#!/usr/bin/env python3
"""
Update deployment status tracking.
Records deployment status and metadata for monitoring and rollback purposes.
"""

import argparse
import json
import sys
import boto3
from datetime import datetime
from typing import Dict, Any, Optional


class DeploymentStatusTracker:
    """Tracks deployment status and metadata."""

    def __init__(self, environment: str):
        self.environment = environment
        self.dynamodb = boto3.resource('dynamodb')
        self.table_name = f'tournament-optimization-{environment}-deployments'

        # Create table if it doesn't exist
        self._ensure_table_exists()

    def _ensure_table_exists(self):
        """Ensure DynamoDB table exists for deployment tracking."""
        try:
            table = self.dynamodb.Table(self.table_name)
            table.load()
        except Exception:
            # Table doesn't exist, create it
            try:
                table = self.dynamodb.create_table(
                    TableName=self.table_name,
                    KeySchema=[
                        {
                            'AttributeName': 'deployment_id',
                            'KeyType': 'HASH'
                        }
                    ],
                    AttributeDefinitions=[
                        {
                            'AttributeName': 'deployment_id',
                            'AttributeType': 'S'
                        }
                    ],
                    BillingMode='PAY_PER_REQUEST',
                    Tags=[
                        {
                            'Key': 'Environment',
                            'Value': self.environment
                        },
                        {
                            'Key': 'Purpose',
                            'Value': 'deployment-tracking'
                        }
                    ]
                )

                # Wait for table to be created
                table.wait_until_exists()
                print(f"Created deployment tracking table: {self.table_name}")

            except Exception as e:
                print(f"Warning: Could not create deployment tracking table: {e}")

    def update_deployment_status(
        self,
        commit: str,
        status: str,
        deployment_strategy: str = 'blue-green',
        image_tag: Optional[str] = None,
        rollback_commit: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update deployment status in tracking table."""

        try:
            table = self.dynamodb.Table(self.table_name)

            deployment_id = f"{self.environment}-{commit[:8]}"
            timestamp = datetime.utcnow().isoformat()

            item = {
                'deployment_id': deployment_id,
                'environment': self.environment,
                'commit': commit,
                'status': status,
                'deployment_strategy': deployment_strategy,
                'timestamp': timestamp,
                'updated_at': timestamp
            }

            if image_tag:
                item['image_tag'] = image_tag

            if rollback_commit:
                item['rollback_commit'] = rollback_commit

            if metadata:
                item['metadata'] = json.dumps(metadata)

            # Add status-specific fields
            if status == 'started':
                item['started_at'] = timestamp
            elif status == 'completed':
                item['completed_at'] = timestamp
            elif status == 'failed':
                item['failed_at'] = timestamp
            elif status == 'rolled_back':
                item['rolled_back_at'] = timestamp

            table.put_item(Item=item)

            print(f"✅ Updated deployment status: {deployment_id} -> {status}")
            return True

        except Exception as e:
            print(f"❌ Failed to update deployment status: {e}")
            return False

    def get_deployment_history(self, limit: int = 10) -> list:
        """Get recent deployment history."""
        try:
            table = self.dynamodb.Table(self.table_name)

            response = table.scan(
                FilterExpression='environment = :env',
                ExpressionAttributeValues={':env': self.environment},
                Limit=limit
            )

            # Sort by timestamp descending
            items = sorted(
                response.get('Items', []),
                key=lambda x: x.get('timestamp', ''),
                reverse=True
            )

            return items[:limit]

        except Exception as e:
            print(f"❌ Failed to get deployment history: {e}")
            return []

    def get_current_deployment(self) -> Optional[Dict[str, Any]]:
        """Get current active deployment."""
        try:
            history = self.get_deployment_history(limit=5)

            # Find the most recent successful deployment
            for deployment in history:
                if deployment.get('status') == 'completed':
                    return deployment

            return None

        except Exception as e:
            print(f"❌ Failed to get current deployment: {e}")
            return None

    def mark_deployment_for_rollback(self, commit: str, reason: str) -> bool:
        """Mark a deployment for rollback."""
        try:
            table = self.dynamodb.Table(self.table_name)
            deployment_id = f"{self.environment}-{commit[:8]}"

            table.update_item(
                Key={'deployment_id': deployment_id},
                UpdateExpression='SET rollback_reason = :reason, rollback_marked_at = :timestamp',
                ExpressionAttributeValues={
                    ':reason': reason,
                    ':timestamp': datetime.utcnow().isoformat()
                }
            )

            print(f"✅ Marked deployment {deployment_id} for rollback: {reason}")
            return True

        except Exception as e:
            print(f"❌ Failed to mark deployment for rollback: {e}")
            return False

    def print_deployment_history(self):
        """Print deployment history in a readable format."""
        history = self.get_deployment_history()

        if not history:
            print("No deployment history found")
            return

        print(f"\n📋 Deployment History - {self.environment.upper()}")
        print("=" * 80)
        print(f"{'Deployment ID':<20} {'Status':<12} {'Commit':<10} {'Strategy':<12} {'Timestamp':<20}")
        print("-" * 80)

        for deployment in history:
            deployment_id = deployment.get('deployment_id', 'N/A')
            status = deployment.get('status', 'N/A')
            commit = deployment.get('commit', 'N/A')[:8]
            strategy = deployment.get('deployment_strategy', 'N/A')
            timestamp = deployment.get('timestamp', 'N/A')[:19]  # Remove microseconds

            # Color code status
            status_icon = {
                'started': '🟡',
                'completed': '✅',
                'failed': '❌',
                'rolled_back': '🔄'
            }.get(status, '⚪')

            print(f"{deployment_id:<20} {status_icon} {status:<10} {commit:<10} {strategy:<12} {timestamp:<20}")

        print("=" * 80)

    def cleanup_old_deployments(self, retention_days: int = 30) -> int:
        """Clean up old deployment records."""
        try:
            from datetime import timedelta

            cutoff_date = datetime.utcnow() - timedelta(days=retention_days)
            cutoff_iso = cutoff_date.isoformat()

            table = self.dynamodb.Table(self.table_name)

            # Scan for old deployments
            response = table.scan(
                FilterExpression='environment = :env AND #ts < :cutoff',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':env': self.environment,
                    ':cutoff': cutoff_iso
                }
            )

            deleted_count = 0

            # Delete old records
            for item in response.get('Items', []):
                table.delete_item(
                    Key={'deployment_id': item['deployment_id']}
                )
                deleted_count += 1

            if deleted_count > 0:
                print(f"✅ Cleaned up {deleted_count} old deployment records")

            return deleted_count

        except Exception as e:
            print(f"❌ Failed to cleanup old deployments: {e}")
            return 0


def main():
    parser = argparse.ArgumentParser(description='Update deployment status')
    parser.add_argument('--environment', required=True, choices=['staging', 'production'],
                       help='Target environment')
    parser.add_argument('--commit', required=True,
                       help='Git commit hash')
    parser.add_argument('--status', required=True,
                       choices=['started', 'completed', 'failed', 'rolled_back'],
                       help='Deployment status')
    parser.add_argument('--strategy', default='blue-green',
                       choices=['blue-green', 'rolling'],
                       help='Deployment strategy')
    parser.add_argument('--image-tag',
                       help='Docker image tag')
    parser.add_argument('--rollback-commit',
                       help='Commit hash being rolled back to')
    parser.add_argument('--metadata',
                       help='Additional metadata as JSON string')
    parser.add_argument('--history', action='store_true',
                       help='Show deployment history')
    parser.add_argument('--cleanup', action='store_true',
                       help='Cleanup old deployment records')

    args = parser.parse_args()

    tracker = DeploymentStatusTracker(args.environment)

    if args.history:
        tracker.print_deployment_history()
        return 0

    if args.cleanup:
        deleted_count = tracker.cleanup_old_deployments()
        return 0

    # Parse metadata if provided
    metadata = None
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid metadata JSON: {e}")
            return 1

    # Update deployment status
    success = tracker.update_deployment_status(
        commit=args.commit,
        status=args.status,
        deployment_strategy=args.strategy,
        image_tag=args.image_tag,
        rollback_commit=args.rollback_commit,
        metadata=metadata
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
