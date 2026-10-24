"""Record how many provider calls each request took (retries and fallbacks)

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-24
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("requests") as batch:
        batch.add_column(sa.Column("attempts", sa.Integer(), nullable=False, server_default="1"))


def downgrade() -> None:
    with op.batch_alter_table("requests") as batch:
        batch.drop_column("attempts")
