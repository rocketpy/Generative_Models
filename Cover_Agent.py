# Cover-Agent - CodiumAI Cover Agent aims to help efficiently increasing code coverage, by automatically generating qualified tests to enhance existing test suites

# https://github.com/Codium-ai/cover-agent


# To install the Python Pip package directly via GitHub run the following command:
# pip install git+https://github.com/Codium-ai/cover-agent.git


# Repository Setup
# poetry install


# Running the Code
"""
cover-agent \
  --source-file-path "<path_to_source_file>" \
  --test-file-path "<path_to_test_file>" \
  --code-coverage-report-path "<path_to_coverage_report>" \
  --test-command "<test_command_to_run>" \
  --test-command-dir "<directory_to_run_test_command>" \
  --coverage-type "<type_of_coverage_report>" \
  --desired-coverage <desired_coverage_between_0_and_100> \
  --max-iterations <max_number_of_llm_iterations> \
  --included-files "<optional_list_of_files_to_include>"
"""
