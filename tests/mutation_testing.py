"""Mutation testing for test quality validation."""

import ast
import random
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import importlib.util


class MutationOperator:
    """Base class for mutation operators."""

    def __init__(self, name: str):
        self.name = name

    def can_mutate(self, node: ast.AST) -> bool:
        """Check if this operator can mutate the given node."""
        raise NotImplementedError

    def mutate(self, node: ast.AST) -> ast.AST:
        """Apply mutation to the node."""
        raise NotImplementedError


class ArithmeticOperatorMutator(MutationOperator):
    """Mutates arithmetic operators (+, -, *, /, etc.)."""

    def __init__(self):
        super().__init__("Arithmetic Operator")
        self.mutations = {
            ast.Add: ast.Sub,
            ast.Sub: ast.Add,
            ast.Mult: ast.Div,
            ast.Div: ast.Mult,
            ast.Mod: ast.Mult,
            ast.Pow: ast.Mult
        }

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.BinOp) and type(node.op) in self.mutations

    def mutate(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.BinOp):
            new_op = self.mutations[type(node.op)]()
            node.op = new_op
        return node


class ComparisonOperatorMutator(MutationOperator):
    """Mutates comparison operators (==, !=, <, >, etc.)."""

    def __init__(self):
        super().__init__("Comparison Operator")
        self.mutations = {
            ast.Eq: ast.NotEq,
            ast.NotEq: ast.Eq,
            ast.Lt: ast.GtE,
            ast.LtE: ast.Gt,
            ast.Gt: ast.LtE,
            ast.GtE: ast.Lt
        }

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in self.mutations

    def mutate(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            old_op_type = type(node.ops[0])
            if old_op_type in self.mutations:
                new_op = self.mutations[old_op_type]()
                node.ops = [new_op]
        return node


class BooleanOperatorMutator(MutationOperator):
    """Mutates boolean operators (and, or)."""

    def __init__(self):
        super().__init__("Boolean Operator")
        self.mutations = {
            ast.And: ast.Or,
            ast.Or: ast.And
        }

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.BoolOp) and type(node.op) in self.mutations

    def mutate(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.BoolOp):
            new_op = self.mutations[type(node.op)]()
            node.op = new_op
        return node


class UnaryOperatorMutator(MutationOperator):
    """Mutates unary operators (not, -, +)."""

    def __init__(self):
        super().__init__("Unary Operator")

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.Not, ast.UAdd, ast.USub))

    def mutate(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                # Remove 'not' operator
                return node.operand
            elif isinstance(node.op, ast.UAdd):
                node.op = ast.USub()
            elif isinstance(node.op, ast.USub):
                node.op = ast.UAdd()
        return node


class ConditionalBoundaryMutator(MutationOperator):
    """Mutates conditional boundaries (< to <=, etc.)."""

    def __init__(self):
        super().__init__("Conditional Boundary")
        self.mutations = {
            ast.Lt: ast.LtE,
            ast.LtE: ast.Lt,
            ast.Gt: ast.GtE,
            ast.GtE: ast.Gt
        }

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in self.mutations

    def mutate(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            old_op_type = type(node.ops[0])
            if old_op_type in self.mutations:
                new_op = self.mutations[old_op_type]()
                node.ops = [new_op]
        return node


class ConstantMutator(MutationOperator):
    """Mutates constants (numbers, strings, booleans)."""

    def __init__(self):
        super().__init__("Constant")

    def can_mutate(self, node: ast.AST) -> bool:
        return isinstance(node, (ast.Constant, ast.Num, ast.Str, ast.NameConstant))

    def mutate(self, node: ast.AST) -> ast.AST:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                node.value = not node.value
            elif isinstance(node.value, int):
                if node.value == 0:
                    node.value = 1
                else:
                    node.value = 0
            elif isinstance(node.value, float):
                if node.value == 0.0:
                    node.value = 1.0
                else:
                    node.value = 0.0
            elif isinstance(node.value, str):
                if node.value == "":
                    node.value = "mutated"
                else:
                    node.value = ""
        return node


class MutationTester:
    """Main mutation testing engine."""

    def __init__(self, source_dir: str = "src", test_dir: str = "tests"):
        self.source_dir = Path(source_dir)
        self.test_dir = Path(test_dir)
        self.operators = [
            ArithmeticOperatorMutator(),
            ComparisonOperatorMutator(),
            BooleanOperatorMutator(),
            UnaryOperatorMutator(),
            ConditionalBoundaryMutator(),
            ConstantMutator()
        ]
        self.mutation_results = []

    def run_mutation_testing(self, target_files: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run mutation testing on specified files or all source files."""
        print("🧬 Starting mutation testing...")

        if target_files:
            files_to_test = [self.source_dir / f for f in target_files]
        else:
            files_to_test = list(self.source_dir.rglob("*.py"))

        total_mutations = 0
        killed_mutations = 0
        survived_mutations = 0

        for source_file in files_to_test:
            if self._should_skip_file(source_file):
                continue

            print(f"Testing mutations in {source_file.relative_to(self.source_dir)}")

            file_results = self._test_file_mutations(source_file)

            total_mutations += file_results['total_mutations']
            killed_mutations += file_results['killed_mutations']
            survived_mutations += file_results['survived_mutations']

            self.mutation_results.append({
                'file': str(source_file.relative_to(self.source_dir)),
                'results': file_results
            })

        # Calculate mutation score
        mutation_score = (killed_mutations / total_mutations * 100) if total_mutations > 0 else 0

        results = {
            'total_mutations': total_mutations,
            'killed_mutations': killed_mutations,
            'survived_mutations': survived_mutations,
            'mutation_score': mutation_score,
            'file_results': self.mutation_results
        }

        self._generate_mutation_report(results)

        return results

    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped for mutation testing."""
        skip_patterns = [
            '__pycache__',
            '.pyc',
            '__init__.py',
            'test_',
            'conftest.py'
        ]

        file_str = str(file_path)
        return any(pattern in file_str for pattern in skip_patterns)

    def _test_file_mutations(self, source_file: Path) -> Dict[str, Any]:
        """Test all possible mutations for a single file."""
        with open(source_file, 'r') as f:
            source_code = f.read()

        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return {
                'total_mutations': 0,
                'killed_mutations': 0,
                'survived_mutations': 0,
                'error': 'Syntax error in source file'
            }

        # Find all possible mutation points
        mutation_points = self._find_mutation_points(tree)

        total_mutations = len(mutation_points)
        killed_mutations = 0
        survived_mutations = 0

        for i, (node, operator) in enumerate(mutation_points):
            print(f"  Testing mutation {i+1}/{total_mutations}: {operator.name}")

            # Create mutated version
            mutated_tree = self._create_mutated_tree(tree, node, operator)

            # Test the mutation
            if self._test_mutation(source_file, mutated_tree):
                killed_mutations += 1
            else:
                survived_mutations += 1

        return {
            'total_mutations': total_mutations,
            'killed_mutations': killed_mutations,
            'survived_mutations': survived_mutations
        }

    def _find_mutation_points(self, tree: ast.AST) -> List[Tuple[ast.AST, MutationOperator]]:
        """Find all nodes that can be mutated."""
        mutation_points = []

        for node in ast.walk(tree):
            for operator in self.operators:
                if operator.can_mutate(node):
                    mutation_points.append((node, operator))

        return mutation_points

    def _create_mutated_tree(self, original_tree: ast.AST, target_node: ast.AST, operator: MutationOperator) -> ast.AST:
        """Create a mutated copy of the AST."""
        # Create a deep copy of the tree
        mutated_tree = ast.parse(ast.unparse(original_tree))

        # Find the corresponding node in the copy and mutate it
        for node in ast.walk(mutated_tree):
            if self._nodes_equivalent(node, target_node):
                operator.mutate(node)
                break

        return mutated_tree

    def _nodes_equivalent(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Check if two AST nodes are equivalent (simplified)."""
        # This is a simplified comparison - in practice, you'd need more sophisticated matching
        return (type(node1) == type(node2) and
                getattr(node1, 'lineno', None) == getattr(node2, 'lineno', None) and
                getattr(node1, 'col_offset', None) == getattr(node2, 'col_offset', None))

    def _test_mutation(self, source_file: Path, mutated_tree: ast.AST) -> bool:
        """Test if a mutation is killed by running tests."""
        # Create temporary file with mutated code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file.write(ast.unparse(mutated_tree))
            temp_file_path = temp_file.name

        try:
            # Backup original file
            backup_path = str(source_file) + '.backup'
            shutil.copy2(source_file, backup_path)

            # Replace original with mutated version
            shutil.copy2(temp_file_path, source_file)

            # Run tests
            test_result = self._run_tests_for_file(source_file)

            # Restore original file
            shutil.copy2(backup_path, source_file)

            # Clean up
            Path(backup_path).unlink()
            Path(temp_file_path).unlink()

            # Mutation is killed if tests fail
            return not test_result

        except Exception as e:
            print(f"    Error testing mutation: {e}")
            # Restore original file in case of error
            backup_path = str(source_file) + '.backup'
            if Path(backup_path).exists():
                shutil.copy2(backup_path, source_file)
                Path(backup_path).unlink()

            if Path(temp_file_path).exists():
                Path(temp_file_path).unlink()

            return False

    def _run_tests_for_file(self, source_file: Path) -> bool:
        """Run tests related to the source file."""
        # Find corresponding test file
        relative_path = source_file.relative_to(self.source_dir)
        test_file_patterns = [
            self.test_dir / "unit" / relative_path.parent / f"test_{relative_path.stem}.py",
            self.test_dir / f"test_{relative_path.stem}.py",
            self.test_dir / "unit" / f"test_{relative_path.stem}.py"
        ]

        test_files = [tf for tf in test_file_patterns if tf.exists()]

        if not test_files:
            # Run all tests if no specific test file found
            cmd = ["python", "-m", "pytest", "-x", "--tb=no", "-q"]
        else:
            # Run specific test files
            cmd = ["python", "-m", "pytest", "-x", "--tb=no", "-q"] + [str(tf) for tf in test_files]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False

    def _generate_mutation_report(self, results: Dict[str, Any]) -> None:
        """Generate mutation testing report."""
        print("\n🧬 MUTATION TESTING REPORT")
        print("=" * 50)
        print(f"Total mutations tested: {results['total_mutations']}")
        print(f"Mutations killed: {results['killed_mutations']}")
        print(f"Mutations survived: {results['survived_mutations']}")
        print(f"Mutation score: {results['mutation_score']:.1f}%")
        print()

        # Quality assessment
        if results['mutation_score'] >= 80:
            print("✅ Excellent test quality (≥80% mutation score)")
        elif results['mutation_score'] >= 60:
            print("⚠️  Good test quality (60-79% mutation score)")
        elif results['mutation_score'] >= 40:
            print("⚠️  Fair test quality (40-59% mutation score)")
        else:
            print("❌ Poor test quality (<40% mutation score)")

        print()
        print("File-by-file results:")
        print("-" * 30)

        for file_result in results['file_results']:
            file_name = file_result['file']
            file_data = file_result['results']

            if file_data['total_mutations'] > 0:
                file_score = (file_data['killed_mutations'] / file_data['total_mutations']) * 100
                print(f"{file_name}: {file_score:.1f}% ({file_data['killed_mutations']}/{file_data['total_mutations']})")
            else:
                print(f"{file_name}: No mutations possible")


def main():
    """Main function to run mutation testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Run mutation testing")
    parser.add_argument("--files", nargs="*", help="Specific files to test")
    parser.add_argument("--source-dir", default="src", help="Source directory")
    parser.add_argument("--test-dir", default="tests", help="Test directory")

    args = parser.parse_args()

    tester = MutationTester(source_dir=args.source_dir, test_dir=args.test_dir)
    results = tester.run_mutation_testing(target_files=args.files)

    # Exit with error code if mutation score is too low
    if results['mutation_score'] < 60:
        print(f"\n❌ Mutation score {results['mutation_score']:.1f}% below threshold (60%)")
        exit(1)
    else:
        print(f"\n✅ Mutation score {results['mutation_score']:.1f}% meets quality threshold")
        exit(0)


if __name__ == "__main__":
    main()
