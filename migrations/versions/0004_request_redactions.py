"""Record how many values were redacted from each request before it reached a provider

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("requests") as batch:
        batch.add_column(sa.Column("redactions", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    with op.batch_alter_table("requests") as batch:
        batch.drop_column("redactions")
