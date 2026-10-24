"""Per-key rate limits

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.add_column(sa.Column("rate_limit_rpm", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("api_keys") as batch:
        batch.drop_column("rate_limit_rpm")
