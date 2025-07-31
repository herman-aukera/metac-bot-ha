"""Automated test coverage reporting and quality gates."""

import pytest
import subprocess
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import xml.etree.ElementTree as ET


class CoverageReporter:
    """Test coverage reporter with quality gates."""

    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.coverage_thresholds = {
            'domain': 90.0,      # Domain layer must have >90% coverage
            'application': 85.0,  # Application layer must have >85% coverage
            'infrastructure': 80.0, # Infrastructure layer must have >80% coverage
            'overall': 85.0      # Overall project must have >85% coverage
        }

    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run comprehensive coverage analysis."""
        print("🔍 Running test coverage analysis...")

        # Run pytest with coverage
        cmd = [
            "python", "-m", "pytest",
            "--cov=src",
            "--cov-report=xml:coverage.xml",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing",
            "--cov-branch",
            "-v"
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.project_root)

            if result.returncode != 0:
                print(f"❌ Test execution failed:")
                print(result.stdout)
                print(result.stderr)
                return {'success': False, 'error': 'Test execution failed'}

            # Parse coverage results
            coverage_data = self._parse_coverage_xml()

            # Generate coverage report
            report = self._generate_coverage_report(coverage_data)

            # Check quality gates
            quality_gates_passed = self._check_quality_gates(coverage_data)

            return {
                'success': True,
                'coverage_data': coverage_data,
                'report': report,
                'quality_gates_passed': quality_gates_passed,
                'test_output': result.stdout
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _parse_coverage_xml(self) -> Dict[str, Any]:
        """Parse coverage XML report."""
        coverage_file = self.project_root / "coverage.xml"

        if not coverage_file.exists():
            raise FileNotFoundError("Coverage XML file not found")

        tree = ET.parse(coverage_file)
        root = tree.getroot()

        # Parse overall coverage
        overall_coverage = float(root.attrib.get('line-rate', 0)) * 100
        branch_coverage = float(root.attrib.get('branch-rate', 0)) * 100

        # Parse package-level coverage
        packages = {}
        for package in root.findall('.//package'):
            package_name = package.attrib['name']
            package_line_rate = float(package.attrib.get('line-rate', 0)) * 100
            package_branch_rate = float(package.attrib.get('branch-rate', 0)) * 100

            # Parse class-level coverage within package
            classes = {}
            for cls in package.findall('.//class'):
                class_name = cls.attrib['name']
                class_line_rate = float(cls.attrib.get('line-rate', 0)) * 100
                classes[class_name] = {
                    'line_coverage': class_line_rate,
                    'filename': cls.attrib.get('filename', '')
                }

            packages[package_name] = {
                'line_coverage': package_line_rate,
                'branch_coverage': package_branch_rate,
                'classes': classes
            }

        return {
            'overall_line_coverage': overall_coverage,
            'overall_branch_coverage': branch_coverage,
            'packages': packages
        }

    def _generate_coverage_report(self, coverage_data: Dict[str, Any]) -> str:
        """Generate human-readable coverage report."""
        report_lines = []
        report_lines.append("📊 TEST COVERAGE REPORT")
        report_lines.append("=" * 50)

        # Overall coverage
        overall_line = coverage_data['overall_line_coverage']
        overall_branch = coverage_data['overall_branch_coverage']

        report_lines.append(f"Overall Line Coverage: {overall_line:.1f}%")
        report_lines.append(f"Overall Branch Coverage: {overall_branch:.1f}%")
        report_lines.append("")

        # Layer-specific coverage
        layer_coverage = self._calculate_layer_coverage(coverage_data)

        report_lines.append("Coverage by Layer:")
        report_lines.append("-" * 20)

        for layer, coverage in layer_coverage.items():
            threshold = self.coverage_thresholds.get(layer, 80.0)
            status = "✅" if coverage >= threshold else "❌"
            report_lines.append(f"{status} {layer.capitalize()}: {coverage:.1f}% (threshold: {threshold:.1f}%)")

        report_lines.append("")

        # Package details
        report_lines.append("Package Coverage Details:")
        report_lines.append("-" * 30)

        for package_name, package_data in coverage_data['packages'].items():
            line_cov = package_data['line_coverage']
            branch_cov = package_data['branch_coverage']
            report_lines.append(f"{package_name}: {line_cov:.1f}% lines, {branch_cov:.1f}% branches")

        return "\n".join(report_lines)

    def _calculate_layer_coverage(self, coverage_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate coverage by architectural layer."""
        layer_coverage = {
            'domain': 0.0,
            'application': 0.0,
            'infrastructure': 0.0
        }

        layer_totals = {layer: {'covered': 0, 'total': 0} for layer in layer_coverage.keys()}

        for package_name, package_data in coverage_data['packages'].items():
            # Determine layer based on package name
            layer = None
            if 'domain' in package_name:
                layer = 'domain'
            elif 'application' in package_name:
                layer = 'application'
            elif 'infrastructure' in package_name:
                layer = 'infrastructure'

            if layer:
                # This is a simplified calculation - in reality you'd need more detailed metrics
                coverage_rate = package_data['line_coverage'] / 100
                estimated_lines = 100  # Placeholder - would need actual line counts

                layer_totals[layer]['covered'] += int(coverage_rate * estimated_lines)
                layer_totals[layer]['total'] += estimated_lines

        # Calculate final percentages
        for layer, totals in layer_totals.items():
            if totals['total'] > 0:
                layer_coverage[layer] = (totals['covered'] / totals['total']) * 100

        return layer_coverage

    def _check_quality_gates(self, coverage_data: Dict[str, Any]) -> Dict[str, bool]:
        """Check if coverage meets quality gate thresholds."""
        layer_coverage = self._calculate_layer_coverage(coverage_data)
        overall_coverage = coverage_data['overall_line_coverage']

        quality_gates = {}

        # Check layer-specific thresholds
        for layer, coverage in layer_coverage.items():
            threshold = self.coverage_thresholds[layer]
            quality_gates[f"{layer}_coverage"] = coverage >= threshold

        # Check overall threshold
        overall_threshold = self.coverage_thresholds['overall']
        quality_gates['overall_coverage'] = overall_coverage >= overall_threshold

        # All gates must pass
        quality_gates['all_gates_passed'] = all(quality_gates.values())

        return quality_gates


def main():
    """Main function to run coverage analysis."""
    reporter = CoverageReporter()
    result = reporter.run_coverage_analysis()

    if result['success']:
        print(result['report'])

        if result['quality_gates_passed']['all_gates_passed']:
            print("\n✅ All quality gates passed!")
            exit(0)
        else:
            print("\n❌ Quality gates failed!")
            failed_gates = [k for k, v in result['quality_gates_passed'].items() if not v and k != 'all_gates_passed']
            print(f"Failed gates: {', '.join(failed_gates)}")
            exit(1)
    else:
        print(f"❌ Coverage analysis failed: {result['error']}")
        exit(1)


if __name__ == "__main__":
    main()
