"""
Database query optimization and indexing strategies.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
from enum import Enum
import hashlib
import json

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of database queries."""
    SELECT = "select"
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    AGGREGATE = "aggregate"


class IndexType(Enum):
    """Types of database indexes."""
    BTREE = "btree"
    HASH = "hash"
    GIN = "gin"
    GIST = "gist"
    COMPOSITE = "composite"


@dataclass
class QueryStats:
    """Statistics for a database query."""
    query_hash: str
    query_type: QueryType
    execution_count: int
    total_time: float
    avg_time: float
    min_time: float
    max_time: float
    last_executed: datetime
    tables_accessed: Set[str]
    columns_accessed: Set[str]
    index_usage: Dict[str, int]

    @property
    def queries_per_second(self) -> float:
        """Calculate queries per second over last hour."""
        # This would need actual timing data in a real implementation
        return self.execution_count / 3600.0


@dataclass
class IndexRecommendation:
    """Recommendation for database index creation."""
    table_name: str
    columns: List[str]
    index_type: IndexType
    estimated_benefit: float
    reason: str
    priority: str  # 'high', 'medium', 'low'
    estimated_size: int  # bytes


@dataclass
class QueryOptimizationSuggestion:
    """Suggestion for query optimization."""
    query_hash: str
    original_query: str
    suggested_query: str
    optimization_type: str
    estimated_improvement: float
    reason: str


class QueryOptimizer:
    """Database query optimizer with performance analysis."""

    def __init__(self, slow_query_threshold: float = 1.0):
        self.slow_query_threshold = slow_query_threshold
        self.query_stats: Dict[str, QueryStats] = {}
        self.query_cache: Dict[str, Any] = {}
        self.index_recommendations: List[IndexRecommendation] = []
        self.optimization_suggestions: List[QueryOptimizationSuggestion] = []

        # Query patterns for optimization
        self.common_patterns = {
            'n_plus_one': [],
            'missing_indexes': [],
            'inefficient_joins': [],
            'unnecessary_columns': []
        }

    def record_query_execution(
        self,
        query: str,
        execution_time: float,
        tables_accessed: Optional[Set[str]] = None,
        columns_accessed: Optional[Set[str]] = None,
        index_usage: Optional[Dict[str, int]] = None
    ) -> None:
        """Record query execution for analysis."""
        query_hash = self._hash_query(query)
        query_type = self._detect_query_type(query)

        if query_hash in self.query_stats:
            # Update existing stats
            stats = self.query_stats[query_hash]
            stats.execution_count += 1
            stats.total_time += execution_time
            stats.avg_time = stats.total_time / stats.execution_count
            stats.min_time = min(stats.min_time, execution_time)
            stats.max_time = max(stats.max_time, execution_time)
            stats.last_executed = datetime.now()

            if tables_accessed:
                stats.tables_accessed.update(tables_accessed)
            if columns_accessed:
                stats.columns_accessed.update(columns_accessed)
            if index_usage:
                for index, count in index_usage.items():
                    stats.index_usage[index] = stats.index_usage.get(index, 0) + count
        else:
            # Create new stats
            self.query_stats[query_hash] = QueryStats(
                query_hash=query_hash,
                query_type=query_type,
                execution_count=1,
                total_time=execution_time,
                avg_time=execution_time,
                min_time=execution_time,
                max_time=execution_time,
                last_executed=datetime.now(),
                tables_accessed=tables_accessed or set(),
                columns_accessed=columns_accessed or set(),
                index_usage=index_usage or {}
            )

        # Log slow queries
        if execution_time > self.slow_query_threshold:
            logger.warning(f"Slow query detected: {execution_time:.3f}s - {query[:100]}...")

    def _hash_query(self, query: str) -> str:
        """Create hash for query normalization."""
        # Normalize query by removing literals and whitespace
        normalized = self._normalize_query(query)
        return hashlib.md5(normalized.encode()).hexdigest()

    def _normalize_query(self, query: str) -> str:
        """Normalize query for pattern matching."""
        # Remove extra whitespace
        normalized = ' '.join(query.split())

        # Replace literals with placeholders
        # This is a simplified version - real implementation would be more sophisticated
        import re

        # Replace string literals
        normalized = re.sub(r"'[^']*'", "'?'", normalized)

        # Replace numeric literals
        normalized = re.sub(r'\b\d+\b', '?', normalized)

        return normalized.lower()

    def _detect_query_type(self, query: str) -> QueryType:
        """Detect the type of SQL query."""
        query_lower = query.lower().strip()

        if query_lower.startswith('select'):
            if any(agg in query_lower for agg in ['count(', 'sum(', 'avg(', 'max(', 'min(']):
                return QueryType.AGGREGATE
            return QueryType.SELECT
        elif query_lower.startswith('insert'):
            return QueryType.INSERT
        elif query_lower.startswith('update'):
            return QueryType.UPDATE
        elif query_lower.startswith('delete'):
            return QueryType.DELETE
        else:
            return QueryType.SELECT  # Default

    def analyze_query_patterns(self) -> Dict[str, List[str]]:
        """Analyze query patterns for optimization opportunities."""
        patterns = {
            'slow_queries': [],
            'frequent_queries': [],
            'missing_indexes': [],
            'n_plus_one_candidates': []
        }

        # Find slow queries
        for query_hash, stats in self.query_stats.items():
            if stats.avg_time > self.slow_query_threshold:
                patterns['slow_queries'].append(query_hash)

        # Find frequent queries
        sorted_by_frequency = sorted(
            self.query_stats.items(),
            key=lambda x: x[1].execution_count,
            reverse=True
        )

        patterns['frequent_queries'] = [
            query_hash for query_hash, stats in sorted_by_frequency[:10]
        ]

        # Detect potential N+1 queries
        self._detect_n_plus_one_queries(patterns)

        # Detect missing indexes
        self._detect_missing_indexes(patterns)

        return patterns

    def _detect_n_plus_one_queries(self, patterns: Dict[str, List[str]]) -> None:
        """Detect potential N+1 query patterns."""
        # Group queries by table access patterns
        table_access_patterns = defaultdict(list)

        for query_hash, stats in self.query_stats.items():
            if stats.query_type == QueryType.SELECT and len(stats.tables_accessed) == 1:
                table = list(stats.tables_accessed)[0]
                table_access_patterns[table].append((query_hash, stats))

        # Look for patterns where many similar queries access the same table
        for table, queries in table_access_patterns.items():
            if len(queries) > 10:  # Threshold for potential N+1
                # Check if queries are executed in quick succession
                recent_queries = [
                    (qh, stats) for qh, stats in queries
                    if (datetime.now() - stats.last_executed).total_seconds() < 60
                ]

                if len(recent_queries) > 5:
                    patterns['n_plus_one_candidates'].extend([qh for qh, _ in recent_queries])

    def _detect_missing_indexes(self, patterns: Dict[str, List[str]]) -> None:
        """Detect queries that might benefit from indexes."""
        # Analyze WHERE clauses and JOIN conditions
        for query_hash, stats in self.query_stats.items():
            if stats.query_type in [QueryType.SELECT, QueryType.AGGREGATE]:
                # Check if query is slow and frequently executed
                if (stats.avg_time > 0.1 and stats.execution_count > 10 and
                    not stats.index_usage):
                    patterns['missing_indexes'].append(query_hash)

    def generate_index_recommendations(self) -> List[IndexRecommendation]:
        """Generate index recommendations based on query analysis."""
        recommendations = []

        # Analyze column usage patterns
        column_usage = defaultdict(lambda: {'count': 0, 'tables': set(), 'query_types': set()})

        for stats in self.query_stats.values():
            for column in stats.columns_accessed:
                column_usage[column]['count'] += stats.execution_count
                column_usage[column]['tables'].update(stats.tables_accessed)
                column_usage[column]['query_types'].add(stats.query_type)

        # Generate recommendations for frequently accessed columns
        for column, usage in column_usage.items():
            if usage['count'] > 100:  # Threshold for recommendation
                for table in usage['tables']:
                    # Determine index type based on usage patterns
                    if QueryType.AGGREGATE in usage['query_types']:
                        index_type = IndexType.BTREE
                        priority = 'high'
                    elif len(usage['query_types']) > 1:
                        index_type = IndexType.BTREE
                        priority = 'medium'
                    else:
                        index_type = IndexType.HASH
                        priority = 'low'

                    recommendation = IndexRecommendation(
                        table_name=table,
                        columns=[column],
                        index_type=index_type,
                        estimated_benefit=self._estimate_index_benefit(column, usage),
                        reason=f"Column accessed {usage['count']} times",
                        priority=priority,
                        estimated_size=self._estimate_index_size(table, [column])
                    )

                    recommendations.append(recommendation)

        # Generate composite index recommendations
        composite_recommendations = self._generate_composite_index_recommendations()
        recommendations.extend(composite_recommendations)

        # Sort by estimated benefit
        recommendations.sort(key=lambda x: x.estimated_benefit, reverse=True)

        self.index_recommendations = recommendations[:20]  # Top 20 recommendations
        return self.index_recommendations

    def _estimate_index_benefit(self, column: str, usage: Dict[str, Any]) -> float:
        """Estimate the benefit of creating an index."""
        # Simplified benefit calculation
        base_benefit = usage['count'] * 0.1  # Base benefit from usage frequency

        # Bonus for aggregate queries
        if QueryType.AGGREGATE in usage['query_types']:
            base_benefit *= 2

        # Bonus for multiple query types
        if len(usage['query_types']) > 1:
            base_benefit *= 1.5

        return base_benefit

    def _estimate_index_size(self, table: str, columns: List[str]) -> int:
        """Estimate the size of an index in bytes."""
        # Simplified size estimation
        base_size = 1024 * 1024  # 1MB base
        column_factor = len(columns) * 512 * 1024  # 512KB per column

        return base_size + column_factor

    def _generate_composite_index_recommendations(self) -> List[IndexRecommendation]:
        """Generate recommendations for composite indexes."""
        recommendations = []

        # Find queries that access multiple columns from the same table
        multi_column_queries = {}

        for query_hash, stats in self.query_stats.items():
            if len(stats.columns_accessed) > 1 and len(stats.tables_accessed) == 1:
                table = list(stats.tables_accessed)[0]
                columns = tuple(sorted(stats.columns_accessed))

                if table not in multi_column_queries:
                    multi_column_queries[table] = {}

                if columns not in multi_column_queries[table]:
                    multi_column_queries[table][columns] = []

                multi_column_queries[table][columns].append(stats)

        # Generate composite index recommendations
        for table, column_groups in multi_column_queries.items():
            for columns, stats_list in column_groups.items():
                total_executions = sum(s.execution_count for s in stats_list)
                avg_time = sum(s.avg_time * s.execution_count for s in stats_list) / total_executions

                if total_executions > 50 and avg_time > 0.1:  # Thresholds
                    recommendation = IndexRecommendation(
                        table_name=table,
                        columns=list(columns),
                        index_type=IndexType.COMPOSITE,
                        estimated_benefit=total_executions * avg_time * 0.5,
                        reason=f"Composite index for {len(columns)} columns used together",
                        priority='high' if total_executions > 200 else 'medium',
                        estimated_size=self._estimate_index_size(table, list(columns))
                    )

                    recommendations.append(recommendation)

        return recommendations

    def generate_query_optimizations(self) -> List[QueryOptimizationSuggestion]:
        """Generate query optimization suggestions."""
        suggestions = []

        # Analyze slow queries for optimization opportunities
        slow_queries = [
            (query_hash, stats) for query_hash, stats in self.query_stats.items()
            if stats.avg_time > self.slow_query_threshold
        ]

        for query_hash, stats in slow_queries:
            # This would contain actual query optimization logic
            # For now, we'll create placeholder suggestions

            if stats.query_type == QueryType.SELECT:
                suggestion = QueryOptimizationSuggestion(
                    query_hash=query_hash,
                    original_query="SELECT * FROM table WHERE condition",
                    suggested_query="SELECT specific_columns FROM table WHERE indexed_condition",
                    optimization_type="column_selection",
                    estimated_improvement=0.3,
                    reason="Avoid SELECT * and use indexed columns"
                )
                suggestions.append(suggestion)

        self.optimization_suggestions = suggestions
        return suggestions

    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive query performance report."""
        # Analyze patterns
        patterns = self.analyze_query_patterns()

        # Generate recommendations
        index_recommendations = self.generate_index_recommendations()
        query_optimizations = self.generate_query_optimizations()

        # Calculate summary statistics
        total_queries = sum(stats.execution_count for stats in self.query_stats.values())
        total_time = sum(stats.total_time for stats in self.query_stats.values())
        avg_query_time = total_time / max(total_queries, 1)

        # Top slow queries
        slow_queries = sorted(
            [(qh, stats) for qh, stats in self.query_stats.items()],
            key=lambda x: x[1].avg_time,
            reverse=True
        )[:10]

        # Most frequent queries
        frequent_queries = sorted(
            [(qh, stats) for qh, stats in self.query_stats.items()],
            key=lambda x: x[1].execution_count,
            reverse=True
        )[:10]

        return {
            'summary': {
                'total_queries': total_queries,
                'unique_queries': len(self.query_stats),
                'total_execution_time': total_time,
                'average_query_time': avg_query_time,
                'slow_query_count': len(patterns['slow_queries'])
            },
            'patterns': patterns,
            'slow_queries': [
                {
                    'query_hash': qh,
                    'avg_time': stats.avg_time,
                    'execution_count': stats.execution_count,
                    'total_time': stats.total_time
                }
                for qh, stats in slow_queries
            ],
            'frequent_queries': [
                {
                    'query_hash': qh,
                    'execution_count': stats.execution_count,
                    'avg_time': stats.avg_time,
                    'queries_per_second': stats.queries_per_second
                }
                for qh, stats in frequent_queries
            ],
            'index_recommendations': [
                {
                    'table': rec.table_name,
                    'columns': rec.columns,
                    'type': rec.index_type.value,
                    'benefit': rec.estimated_benefit,
                    'priority': rec.priority,
                    'reason': rec.reason
                }
                for rec in index_recommendations
            ],
            'optimization_suggestions': [
                {
                    'query_hash': opt.query_hash,
                    'type': opt.optimization_type,
                    'improvement': opt.estimated_improvement,
                    'reason': opt.reason
                }
                for opt in query_optimizations
            ]
        }


class IndexManager:
    """Manages database indexes for optimal performance."""

    def __init__(self, query_optimizer: QueryOptimizer):
        self.query_optimizer = query_optimizer
        self.existing_indexes: Dict[str, List[str]] = {}  # table -> list of indexes
        self.index_usage_stats: Dict[str, int] = {}
        self.maintenance_schedule: Dict[str, datetime] = {}

    def register_existing_index(self, table: str, index_name: str, columns: List[str]) -> None:
        """Register an existing database index."""
        if table not in self.existing_indexes:
            self.existing_indexes[table] = []

        index_info = f"{index_name}({','.join(columns)})"
        self.existing_indexes[table].append(index_info)

        logger.info(f"Registered index: {table}.{index_info}")

    def record_index_usage(self, index_name: str) -> None:
        """Record index usage for statistics."""
        self.index_usage_stats[index_name] = self.index_usage_stats.get(index_name, 0) + 1

    def analyze_index_effectiveness(self) -> Dict[str, Any]:
        """Analyze the effectiveness of existing indexes."""
        analysis = {
            'unused_indexes': [],
            'heavily_used_indexes': [],
            'maintenance_needed': []
        }

        # Find unused indexes
        for table, indexes in self.existing_indexes.items():
            for index in indexes:
                usage_count = self.index_usage_stats.get(index, 0)
                if usage_count == 0:
                    analysis['unused_indexes'].append({
                        'table': table,
                        'index': index,
                        'usage_count': usage_count
                    })
                elif usage_count > 1000:
                    analysis['heavily_used_indexes'].append({
                        'table': table,
                        'index': index,
                        'usage_count': usage_count
                    })

        # Check maintenance schedule
        current_time = datetime.now()
        for index, last_maintenance in self.maintenance_schedule.items():
            if (current_time - last_maintenance).days > 30:  # 30 days threshold
                analysis['maintenance_needed'].append({
                    'index': index,
                    'last_maintenance': last_maintenance.isoformat(),
                    'days_since': (current_time - last_maintenance).days
                })

        return analysis

    def get_index_recommendations(self) -> List[Dict[str, Any]]:
        """Get index recommendations with implementation details."""
        recommendations = self.query_optimizer.generate_index_recommendations()

        detailed_recommendations = []
        for rec in recommendations:
            # Check if similar index already exists
            existing_similar = self._find_similar_indexes(rec.table_name, rec.columns)

            detailed_rec = {
                'table': rec.table_name,
                'columns': rec.columns,
                'type': rec.index_type.value,
                'estimated_benefit': rec.estimated_benefit,
                'priority': rec.priority,
                'reason': rec.reason,
                'estimated_size_mb': rec.estimated_size / (1024 * 1024),
                'similar_existing': existing_similar,
                'sql_command': self._generate_create_index_sql(rec)
            }

            detailed_recommendations.append(detailed_rec)

        return detailed_recommendations

    def _find_similar_indexes(self, table: str, columns: List[str]) -> List[str]:
        """Find existing indexes that are similar to the proposed one."""
        if table not in self.existing_indexes:
            return []

        similar = []
        column_set = set(columns)

        for index in self.existing_indexes[table]:
            # Extract columns from index info (simplified parsing)
            if '(' in index and ')' in index:
                index_columns = index.split('(')[1].split(')')[0].split(',')
                index_column_set = set(col.strip() for col in index_columns)

                # Check for overlap
                if column_set.intersection(index_column_set):
                    similar.append(index)

        return similar

    def _generate_create_index_sql(self, recommendation: IndexRecommendation) -> str:
        """Generate SQL command to create the recommended index."""
        index_name = f"idx_{recommendation.table_name}_{'_'.join(recommendation.columns)}"
        columns_str = ', '.join(recommendation.columns)

        if recommendation.index_type == IndexType.BTREE:
            return f"CREATE INDEX {index_name} ON {recommendation.table_name} USING BTREE ({columns_str});"
        elif recommendation.index_type == IndexType.HASH:
            return f"CREATE INDEX {index_name} ON {recommendation.table_name} USING HASH ({columns_str});"
        elif recommendation.index_type == IndexType.COMPOSITE:
            return f"CREATE INDEX {index_name} ON {recommendation.table_name} ({columns_str});"
        else:
            return f"CREATE INDEX {index_name} ON {recommendation.table_name} ({columns_str});"


# Global optimizer instances
query_optimizer = QueryOptimizer()
index_manager = IndexManager(query_optimizer)
