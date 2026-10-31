"""Начальная схема базы данных.

Revision ID: 0001
Revises:
Create Date: 2026-10-31 17:57:57.170852

"""
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = '0001'
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table('fsm_state',
    sa.Column('key', sa.String(length=256), nullable=False),
    sa.Column('state', sa.String(length=128), nullable=True),
    sa.Column('data', sa.JSON(), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('key')
    )
    op.create_table('users',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('telegram_user_id', sa.BigInteger(), nullable=False),
    sa.Column('display_name', sa.String(length=128), nullable=True),
    sa.Column('username', sa.String(length=64), nullable=True),
    sa.Column('last_seen_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('is_active', sa.Boolean(), server_default=sa.text('true'), nullable=False),
    sa.Column('blocked_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('dreams_total', sa.Integer(), server_default='0', nullable=False),
    sa.Column('qa_answers_total', sa.Integer(), server_default='0', nullable=False),
    sa.Column('locale', sa.String(length=8), server_default='ru', nullable=False),
    sa.Column('acquisition_source', sa.String(length=64), nullable=True),
    sa.Column('access_tier', sa.String(length=16), server_default='new', nullable=False),
    sa.Column('access_tier_until', sa.DateTime(timezone=True), nullable=True),
    sa.Column('spend_reset_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('paywall_shown_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('interview_passed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('tier_before_interview', sa.String(length=16), nullable=True),
    sa.Column('profile_life_context', sa.Text(), nullable=True),
    sa.Column('profile_experience_level', sa.Enum('newbie', 'some_experience', 'in_therapy', name='experience_level'), nullable=True),
    sa.Column('onboarding_completed_at', sa.DateTime(timezone=True), nullable=True),
    sa.Column('menu_version', sa.Integer(), server_default='0', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_users_telegram_user_id'), ['telegram_user_id'], unique=True)

    op.create_table('analytics_events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('telegram_user_id', sa.BigInteger(), nullable=False),
    sa.Column('event_type', sa.String(length=48), nullable=False),
    sa.Column('occurred_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('event_day', sa.Date(), nullable=False),
    sa.Column('event_metadata', sa.JSON(), nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('analytics_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_analytics_events_event_day'), ['event_day'], unique=False)
        batch_op.create_index(batch_op.f('ix_analytics_events_event_type'), ['event_type'], unique=False)
        batch_op.create_index(batch_op.f('ix_analytics_events_occurred_at'), ['occurred_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_analytics_events_telegram_user_id'), ['telegram_user_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_analytics_events_user_id'), ['user_id'], unique=False)

    op.create_table('dreams',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('recorded_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('transcription', sa.String(), nullable=False),
    sa.Column('raw_audio_file_id', sa.String(length=256), nullable=True),
    sa.Column('title', sa.String(length=256), nullable=True),
    sa.Column('status', sa.Enum('recorded', 'in_analysis', 'analyzed', 'fully_explored', name='dream_status'), server_default='recorded', nullable=False),
    sa.Column('kind', sa.String(length=24), server_default='dream', nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('dreams', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_dreams_kind'), ['kind'], unique=False)
        batch_op.create_index(batch_op.f('ix_dreams_user_id'), ['user_id'], unique=False)

    op.create_table('sessions',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dream_id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('type', sa.Enum('quick', 'deep', name='session_type'), nullable=False),
    sa.Column('status', sa.Enum('in_progress', 'paused', 'completed', 'safety_paused', name='session_status'), server_default='in_progress', nullable=False),
    sa.Column('per_session_context', sa.String(length=2048), nullable=True),
    sa.Column('is_onboarding_session', sa.Boolean(), server_default=sa.text('false'), nullable=False),
    sa.Column('request_branch', sa.Enum('feel_state', 'decision', 'conflict', 'calm', 'new_about_self', 'unclear', 'custom', name='request_branch'), nullable=True),
    sa.Column('target_image_id', sa.Integer(), nullable=True),
    sa.Column('current_phase', sa.Enum('p2_focus', 'p3_feel', 'self_gate', 'unblending', 'p4', 'p5', 'p6', name='session_phase'), nullable=True),
    sa.Column('p4_substep', sa.Enum('embodiment', 'origin', 'role', 'attitude', name='p4_substep'), nullable=True),
    sa.Column('pivot_count', sa.Integer(), server_default='0', nullable=False),
    sa.Column('started_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_active_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('finished_at', sa.DateTime(timezone=True), nullable=True),
    sa.ForeignKeyConstraint(['dream_id'], ['dreams.id'], ),
    sa.ForeignKeyConstraint(['target_image_id'], ['dream_images.id'], name='fk_sessions_target_image', use_alter=True),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_sessions_dream_id'), ['dream_id'], unique=False)
        batch_op.create_index(batch_op.f('ix_sessions_user_id'), ['user_id'], unique=False)

    op.create_table('dream_images',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('dream_id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=256), nullable=False),
    sa.Column('ifs_category', sa.Enum('manager', 'firefighter', 'exile', 'unclear', name='ifs_category'), nullable=False),
    sa.Column('shape', sa.Enum('human-like', 'animal', 'object', 'element', 'phenomenon', 'sensation', 'abstract', name='image_shape'), nullable=False),
    sa.Column('analyzed_in_session_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['analyzed_in_session_id'], ['sessions.id'], ),
    sa.ForeignKeyConstraint(['dream_id'], ['dreams.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('dream_images', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_dream_images_dream_id'), ['dream_id'], unique=False)

    op.create_table('llm_usage_events',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=True),
    sa.Column('session_id', sa.Integer(), nullable=True),
    sa.Column('service', sa.Enum('openrouter', 'whisper', name='llm_service'), nullable=False),
    sa.Column('model', sa.String(length=64), nullable=False),
    sa.Column('usage_metric', sa.Enum('input_tokens', 'output_tokens', 'audio_seconds', name='usage_metric'), nullable=False),
    sa.Column('usage_amount', sa.Integer(), nullable=False),
    sa.Column('cost_usd', sa.Numeric(precision=10, scale=6), nullable=False),
    sa.Column('prompt_version', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('llm_usage_events', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_llm_usage_events_session_id'), ['session_id'], unique=False)

    op.create_table('session_messages',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('session_id', sa.Integer(), nullable=False),
    sa.Column('direction', sa.Enum('user_to_bot', 'bot_to_user', name='session_message_direction'), nullable=False),
    sa.Column('content_type', sa.Enum('text', 'voice', name='session_message_content_type'), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('audio_file_id', sa.String(length=256), nullable=True),
    sa.Column('llm_role_used', sa.Enum('generator', 'classifier', name='llm_role'), nullable=True),
    sa.Column('llm_phase_label', sa.String(length=64), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
    sa.PrimaryKeyConstraint('id')
    )

    op.create_table('session_state',
    sa.Column('session_id', sa.Integer(), nullable=False),
    sa.Column('data', sa.JSON(), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['session_id'], ['sessions.id'], ),
    sa.PrimaryKeyConstraint('session_id')
    )
    op.create_table('user_memory_items',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('user_id', sa.Integer(), nullable=False),
    sa.Column('kind', sa.String(length=24), nullable=False),
    sa.Column('text', sa.String(length=500), nullable=False),
    sa.Column('entity_key', sa.String(length=64), nullable=True),
    sa.Column('realm', sa.String(length=16), nullable=False),
    sa.Column('source', sa.String(length=24), nullable=False),
    sa.Column('source_session_id', sa.Integer(), nullable=True),
    sa.Column('dream_id', sa.Integer(), nullable=True),
    sa.Column('status', sa.String(length=16), server_default='active', nullable=False),
    sa.Column('superseded_by_id', sa.Integer(), nullable=True),
    sa.Column('confidence', sa.SmallInteger(), server_default='1', nullable=False),
    sa.Column('occurred_at', sa.Date(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('last_confirmed_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.ForeignKeyConstraint(['dream_id'], ['dreams.id'], ),
    sa.ForeignKeyConstraint(['source_session_id'], ['sessions.id'], ),
    sa.ForeignKeyConstraint(['superseded_by_id'], ['user_memory_items.id'], ),
    sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    with op.batch_alter_table('user_memory_items', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_user_memory_items_kind'), ['kind'], unique=False)
        batch_op.create_index(batch_op.f('ix_user_memory_items_user_id'), ['user_id'], unique=False)

    # sessions <-> dream_images ссылаются друг на друга, поэтому этот ключ
    # добавляется после создания обеих таблиц (в моделях его нет).
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.create_foreign_key(
            "fk_sessions_target_image", "dream_images", ["target_image_id"], ["id"]
        )

    # Составные индексы под частые выборки (админка, лимиты, память, аналитика).
    op.create_index('ix_analytics_events_tg_type_time', 'analytics_events', ['telegram_user_id', 'event_type', 'occurred_at'])
    op.create_index('ix_analytics_events_type_day', 'analytics_events', ['event_type', 'event_day'])
    op.create_index('ix_llm_usage_events_user_id_created_at', 'llm_usage_events', ['user_id', 'created_at'])
    op.create_index('ix_session_messages_session_created', 'session_messages', ['session_id', 'created_at'])
    op.create_index('ix_sessions_user_last_active', 'sessions', ['user_id', 'last_active_at'])
    op.create_index('ix_user_memory_items_user_entity', 'user_memory_items', ['user_id', 'entity_key'])
    op.create_index('ix_user_memory_items_user_status_kind', 'user_memory_items', ['user_id', 'status', 'kind'])


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("sessions") as batch_op:
        batch_op.drop_constraint("fk_sessions_target_image", type_="foreignkey")
    op.drop_index('ix_analytics_events_tg_type_time', table_name='analytics_events')
    op.drop_index('ix_analytics_events_type_day', table_name='analytics_events')
    op.drop_index('ix_llm_usage_events_user_id_created_at', table_name='llm_usage_events')
    op.drop_index('ix_session_messages_session_created', table_name='session_messages')
    op.drop_index('ix_sessions_user_last_active', table_name='sessions')
    op.drop_index('ix_user_memory_items_user_entity', table_name='user_memory_items')
    op.drop_index('ix_user_memory_items_user_status_kind', table_name='user_memory_items')
    with op.batch_alter_table('user_memory_items', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_user_memory_items_user_id'))
        batch_op.drop_index(batch_op.f('ix_user_memory_items_kind'))

    op.drop_table('user_memory_items')
    op.drop_table('session_state')

    op.drop_table('session_messages')
    with op.batch_alter_table('llm_usage_events', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_llm_usage_events_session_id'))

    op.drop_table('llm_usage_events')
    with op.batch_alter_table('dream_images', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_dream_images_dream_id'))

    op.drop_table('dream_images')
    with op.batch_alter_table('sessions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_sessions_user_id'))
        batch_op.drop_index(batch_op.f('ix_sessions_dream_id'))

    op.drop_table('sessions')
    with op.batch_alter_table('dreams', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_dreams_user_id'))
        batch_op.drop_index(batch_op.f('ix_dreams_kind'))

    op.drop_table('dreams')
    with op.batch_alter_table('analytics_events', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_analytics_events_user_id'))
        batch_op.drop_index(batch_op.f('ix_analytics_events_telegram_user_id'))
        batch_op.drop_index(batch_op.f('ix_analytics_events_occurred_at'))
        batch_op.drop_index(batch_op.f('ix_analytics_events_event_type'))
        batch_op.drop_index(batch_op.f('ix_analytics_events_event_day'))

    op.drop_table('analytics_events')
    with op.batch_alter_table('users', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_users_telegram_user_id'))

    op.drop_table('users')
    op.drop_table('fsm_state')
