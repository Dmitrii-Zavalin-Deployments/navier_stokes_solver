# ==============================================================================
# Test Suite Name: test_validate_schema.py
# Description: Literate test suite verifying schema validation utility behavior,
#              covering successful compliance audits, constitution violations, 
#              file access failures, and parsing errors.
# ==============================================================================

# LITERATE TESTING NARRATIVE & GOVERNING BEHAVIOR:
# ------------------------------------------------------------------------------
# The schema validation utility ensures that input configurations strictly conform 
# to the solver's structural schema without relying on implicit defaults. 
# This test suite validates all execution branches of the main() validation pipeline:
# 1. Successful validation and compliance reporting.
# 2. Schema ValidationError handling (Constitution Violations).
# 3. File system access exceptions (OSError).
# 4. JSON parsing failures (JSONDecodeError).
# 5. Unexpected runtime exceptions (TypeError).
# ------------------------------------------------------------------------------

import json
from unittest.mock import mock_open, patch

from jsonschema import ValidationError

from src.utils.validate_schema import main


def test_main_success(capsys):
    """
    Scenario 1: Successful Schema Validation
    - Mocks file opening with valid JSON contents.
    - Verifies that validation passes without triggering errors or exit codes.
    - Asserts that the success message is correctly output to stdout.
    """
    # We mock file opening and JSON loading with a valid schema/config structure:
    with patch("builtins.open", mock_open(read_data="{}")), \
         patch("json.load", return_value={"type": "object"}), \
         patch("jsonschema.validate") as mock_validate, \
         patch("sys.exit") as mock_exit:
        
        # We execute the validation utility's main entry point:
        main()
        
        # We assert that validation was invoked and no termination occurred:
        mock_validate.assert_called_once()
        mock_exit.assert_not_called()
        
        # We capture standard output and verify the compliance success notice:
        captured = capsys.readouterr()
        assert "Schema Compliance Audit PASSED" in captured.out


def test_main_validation_error(capsys):
    """
    Scenario 2: Schema Validation Failure (Constitution Violation)
    - Mocks a jsonschema.ValidationError exception during validation.
    - Verifies that the process terminates with exit code 1.
    - Asserts that the constitution violation diagnostic is printed to stderr.
    """
    # We simulate a validation failure due to missing properties:
    with patch("builtins.open", mock_open(read_data="{}")), \
         patch("json.load", return_value={}), \
         patch("jsonschema.validate", side_effect=ValidationError("Missing required property 'nx'")), \
         patch("sys.exit") as mock_exit:
        
        main()
        
        # We assert that the system terminated with error status 1:
        mock_exit.assert_called_once_with(1)
        
        # We capture standard error and verify the violation notice:
        captured = capsys.readouterr()
        assert "CONSTITUTION VIOLATION" in captured.err
        assert "Missing required property 'nx'" in captured.err


def test_main_os_error(capsys):
    """
    Scenario 3: File System Access Exception (OSError)
    - Mocks an OSError when attempting to access schema or configuration files.
    - Verifies that the process terminates gracefully with exit code 1.
    - Asserts that the critical file error is logged to stderr.
    """
    # We simulate an OS-level file access failure:
    with patch("builtins.open", side_effect=OSError("No such file or directory")), \
         patch("sys.exit") as mock_exit:
        
        main()
        
        # We assert termination with error status 1:
        mock_exit.assert_called_once_with(1)
        
        # We capture standard error and verify the critical message:
        captured = capsys.readouterr()
        assert "CRITICAL ERROR" in captured.err
        assert "No such file or directory" in captured.err


def test_main_json_decode_error(capsys):
    """
    Scenario 4: Malformed JSON Syntax (JSONDecodeError)
    - Mocks a JSONDecodeError caused by malformed input or schema files.
    - Verifies that the process catches the error and exits with code 1.
    """
    # We simulate malformed JSON contents:
    with patch("builtins.open", mock_open(read_data="invalid json data")), \
         patch("json.load", side_effect=json.JSONDecodeError("Expecting value", "invalid json data", 0)), \
         patch("sys.exit") as mock_exit:
        
        main()
        
        # We assert termination with error status 1:
        mock_exit.assert_called_once_with(1)
        
        # We verify the critical error log in stderr:
        captured = capsys.readouterr()
        assert "CRITICAL ERROR" in captured.err


def test_main_type_error(capsys):
    """
    Scenario 5: Unexpected Runtime Exceptions (TypeError/ValueError)
    - Mocks an unexpected TypeError during execution.
    - Verifies that the process handles the exception safely and terminates with code 1.
    """
    # We simulate an unexpected type error during validation:
    with patch("builtins.open", mock_open(read_data="{}")), \
         patch("json.load", return_value={}), \
         patch("jsonschema.validate", side_effect=TypeError("Unsupported type encountered")), \
         patch("sys.exit") as mock_exit:
        
        main()
        
        # We assert termination with error status 1:
        mock_exit.assert_called_once_with(1)
        
        # We verify the critical error details in stderr:
        captured = capsys.readouterr()
        assert "CRITICAL ERROR" in captured.err
        assert "Unsupported type encountered" in captured.err