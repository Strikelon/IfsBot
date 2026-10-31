from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import structlog
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from sqlalchemy import delete, select, update

from adapters.telegram.interview_notice import notify_interview_granted
from admin.deps import templates
from admin.queries import subscriber_rows, user_card_summary
from admin.utils import (
    access_badge,
    access_line,
    avatar_color,
    avatar_initials,
    format_relative_date,
    format_usd,
    interview_badge,
    paywall_badge,
)
from data.models import (
    AnalyticsEvent,
    Dream,
    DreamImage,
    FsmState,
    LlmUsageEvent,
    Session,
    SessionMessage,
    SessionState,
    User,
    UserMemoryItem,
)
from domain.access import (
    TIER_ACTIONS,
    apply_tier_change,
    is_exempt,
    normalize_exempt,
)

router = APIRouter()
log = structlog.get_logger(__name__)

_AVATAR_TTL_SECONDS = 24 * 3600
# Negative cache is short: a user who deleted the bot has no photo, but once they
# re-subscribe their avatar should reappear quickly (not stay stuck for a day).
_AVATAR_MISS_TTL_SECONDS = 600


@router.get("/users/{telegram_user_id}/avatar")
async def user_avatar(request: Request, telegram_user_id: int):
    settings = request.app.state.settings
    bot = request.app.state.bot
    cache_dir = Path(settings.admin_media_dir) / "avatars"
    cache_file = cache_dir / f"{telegram_user_id}.jpg"
    miss_marker = cache_dir / f"{telegram_user_id}.none"

    now = time.time()
    # fresh positive cache
    if cache_file.exists() and (now - cache_file.stat().st_mtime) < _AVATAR_TTL_SECONDS:
        return Response(content=cache_file.read_bytes(), media_type="image/jpeg")
    # fresh negative cache -> 404 (don't re-hit Telegram every request; short TTL
    # so re-subscribed users' avatars reappear quickly)
    if miss_marker.exists() and (now - miss_marker.stat().st_mtime) < _AVATAR_MISS_TTL_SECONDS:
        raise HTTPException(status_code=404)

    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        photos = await bot.get_user_profile_photos(telegram_user_id, limit=1)
        if not photos.total_count:
            miss_marker.write_bytes(b"")
            raise HTTPException(status_code=404)
        # photos.photos[0] is a list of PhotoSize (ascending); take the largest.
        file_id = photos.photos[0][-1].file_id
        tg_file = await bot.get_file(file_id)
        buf = await bot.download_file(tg_file.file_path)
        data = buf.read()
        cache_file.write_bytes(data)
        return Response(content=data, media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 — any Telegram/download failure -> fallback to initials
        miss_marker.write_bytes(b"")
        raise HTTPException(status_code=404) from None


@router.get("/subscribers", response_class=HTMLResponse)
async def subscribers(
    request: Request,
    search: str | None = None,
    sort: str = "created",
    desc: bool = True,
    page: int = 1,
    per_page: int = 50,
):
    factory = request.app.state.session_factory
    async with factory() as session:
        result = await subscriber_rows(
            session, search=search, sort=sort, page=page, per_page=per_page, descending=desc
        )
    return templates.TemplateResponse(
        request,
        "subscribers.html",
        {
            "rows": result.items,
            "total": result.total,
            "page": page,
            "per_page": per_page,
            "search": search or "",
            "sort": sort,
            "desc": desc,
            "fmt_usd": format_usd,
            "fmt_date": format_relative_date,
            "initials": avatar_initials,
            "color": avatar_color,
            "access_badge": access_badge,
            "paywall_badge": paywall_badge,
            "interview_badge": interview_badge,
        },
    )


@router.get("/users/{telegram_user_id}", response_class=HTMLResponse)
async def user_card(request: Request, telegram_user_id: int):
    factory = request.app.state.session_factory
    async with factory() as session:
        summary = await user_card_summary(session, telegram_user_id=telegram_user_id)
        if summary is None:
            raise HTTPException(status_code=404)

    settings = request.app.state.settings
    exempt = is_exempt(summary.username, normalize_exempt(settings.spend_limit_exempt_usernames))
    access = access_line(
        summary.access_tier,
        summary.access_tier_until,
        exempt=exempt,
        early_limit=settings.user_spend_limit_usd,
        new_limit=settings.new_user_spend_limit_usd,
    )
    return templates.TemplateResponse(
        request,
        "user_detail.html",
        {
            "u": summary,
            "fmt_usd": format_usd,
            "fmt_date": format_relative_date,
            "initials": avatar_initials,
            "color": avatar_color,
            "access_badge": access_badge,
            "paywall_badge": paywall_badge,
            "interview_badge": interview_badge,
            "access": access,
        },
    )


@router.delete("/api/users/{telegram_user_id}")
async def user_delete_api(request: Request, telegram_user_id: int):
    factory = request.app.state.session_factory
    async with factory() as session:
        user = (
            await session.execute(select(User).where(User.telegram_user_id == telegram_user_id))
        ).scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404)

        session_ids = (
            await session.execute(select(Session.id).where(Session.user_id == user.id))
        ).scalars().all()
        dream_ids = (
            await session.execute(select(Dream.id).where(Dream.user_id == user.id))
        ).scalars().all()

        # 1. Preserve billing audit: detach usage events from user (keep rows).
        await session.execute(
            update(LlmUsageEvent)
            .where(LlmUsageEvent.user_id == user.id)
            .values(user_id=None)
        )
        # 2. Sessions are about to be deleted — drop the usage->session FK too,
        #    so the preserved billing rows don't dangle.
        if session_ids:
            await session.execute(
                update(LlmUsageEvent)
                .where(LlmUsageEvent.session_id.in_(session_ids))
                .values(session_id=None)
            )
            # 3. Session children (FK -> sessions.id).
            await session.execute(
                delete(SessionMessage).where(SessionMessage.session_id.in_(session_ids))
            )
            await session.execute(
                delete(SessionState).where(SessionState.session_id.in_(session_ids))
            )
            # 4. Break the sessions <-> dream_images cycle: null sessions.target_image_id
            #    before deleting dream_images.
            await session.execute(
                update(Session)
                .where(Session.id.in_(session_ids))
                .values(target_image_id=None)
            )
        # 5. dream_images (FK -> dreams.id and -> sessions.id) before both parents.
        if dream_ids:
            await session.execute(
                delete(DreamImage).where(DreamImage.dream_id.in_(dream_ids))
            )
        # 6. memory items (FK -> users, sessions, dreams — no cascade): must go
        #    before sessions/dreams, иначе source_session_id/dream_id ронял бы
        #    delete(Session) IntegrityError'ом (как analytics_events до фикса).
        await session.execute(
            delete(UserMemoryItem).where(UserMemoryItem.user_id == user.id)
        )
        # 7. sessions (FK -> users, dreams).
        await session.execute(delete(Session).where(Session.user_id == user.id))
        # 8. dreams (FK -> users).
        await session.execute(delete(Dream).where(Dream.user_id == user.id))
        # 9. analytics events (FK -> users, added in mig 0013 — no cascade).
        await session.execute(
            delete(AnalyticsEvent).where(AnalyticsEvent.user_id == user.id)
        )
        # 10. the user.
        await session.execute(delete(User).where(User.id == user.id))
        # 11. FSM-состояние: FK на users нет (PK — сериализованный aiogram
        #     StorageKey "bot_id:chat_id:user_id:thread:business:destiny"), поэтому
        #     delete не падал, а тихо оставлял строку. Юзер после удаления возвращался
        #     в свой прежний шаг вместо чистого онбординга. Совпадение по user_id
        #     в середине ключа: bot_id стоит первым (без ведущего ':'), а chat_id
        #     приватного чата равен самому tg_user_id — чужие ключи не задеваются.
        await session.execute(
            delete(FsmState).where(FsmState.key.like(f"%:{telegram_user_id}:%"))
        )
        await session.commit()
    return JSONResponse({"ok": True})


@router.post("/api/users/{telegram_user_id}/tier")
async def user_set_tier(request: Request, telegram_user_id: int):
    """Сменить статус: обычный тестер / интервью (безлимит на 2 календарных месяца)."""
    payload = await request.json()
    action = (payload or {}).get("tier")
    if action not in TIER_ACTIONS:
        raise HTTPException(status_code=400, detail="unknown tier")

    factory = request.app.state.session_factory
    async with factory() as session:
        user = (
            await session.execute(select(User).where(User.telegram_user_id == telegram_user_id))
        ).scalar_one_or_none()
        if user is None:
            raise HTTPException(status_code=404)

        previous = user.access_tier
        apply_tier_change(user, action, now=datetime.now(UTC))
        tier, until = user.access_tier, user.access_tier_until
        await session.commit()

    # Оповещение — ПОСЛЕ коммита и только на выдачу интервью: снятие статуса
    # юзера не касается. Недоставка не влияет на результат запроса, тир уже сменён.
    notified = False
    if action == "interview":
        async with factory() as session:
            notified = await notify_interview_granted(
                bot=request.app.state.bot,
                session=session,
                telegram_user_id=telegram_user_id,
                until=until,
            )
            await session.commit()

    log.info(
        "admin.tier_changed",
        telegram_user_id=telegram_user_id,
        tier_from=previous,
        tier_to=tier,
        until=until.isoformat() if until else None,
        notified=notified,
    )
    return JSONResponse(
        {
            "ok": True,
            "tier": tier,
            "until": until.isoformat() if until else None,
            "notified": notified,
        }
    )
