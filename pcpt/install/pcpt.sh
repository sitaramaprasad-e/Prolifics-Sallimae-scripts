#!/bin/bash

# Shell script to run the PCPT program in a Docker container

# Get the current directory where the script is executed
CURRENT_DIR=$(pwd)

# Add env var for passing current directory into container
CURRENT_PATH_OVERRIDE="$CURRENT_DIR"


# Override the PROGRAM_PATH with an environment variable
PROGRAM_PATH_OVERRIDE="/app"  # This is the path inside the container
LOG_PATH_OVERRIDE="/log"  # This is the path inside the container
CONFIG_PATH_OVERRIDE="/config"  # This is the path inside the container
PROMPTS_PATH_OVERRIDE="/app/prompts"  # This is the path inside the container
HINTS_PATH_OVERRIDE="/app/hints"  # This is the path inside the container
FILTERS_PATH_OVERRIDE="/app/filters"  # This is the path inside the container

# Optional PlantUML env passthrough (CLI flags or host env)
PLANTUML_SERVER_ENV=""
PLANTUML_TIMEOUT_ENV=""
PLANTUML_RETRIES_ENV=""
ENV_VARS=()

# Initialize arrays for the arguments and the volumes
ARGS=()
VOLUMES=()
OUTPUT_PROVIDED=false  # Flag to track if -o or --output is provided

# Ensure ~/.aws directory exists
if [ ! -d "$HOME/.aws" ]; then
    mkdir -p "$HOME/.aws"
fi

# Function to map paths to the correct volume inside the container
map_path_to_volume() {
    local path=$1
    local container_path=$2

    # Convert the path to an absolute path if it's relative
    local absolute_path=$(realpath "$path")
    #echo "path: $path"
    #echo "container path: $container_path"
    #echo "absolute path: $absolute_path"

    if [[ -z "$absolute_path" ]]; then
        absolute_path="."  # Set to current directory if FOLDER_PATH is empty
        #echo "settg to current because -z"
    fi

    local volume_mapping="-v ${absolute_path}:${container_path}"
    
    # Check if the volume mapping already exists in VOLUMES
    for existing_volume in "${VOLUMES[@]}"; do
        if [[ "$existing_volume" == "$volume_mapping" ]]; then
            # Volume already exists, so return without adding it
            return
        fi
    done

    # Add the volume mapping if it does not already exist
    VOLUMES+=("$volume_mapping")
}

replace_folder_in_path() {
    local original_path="$1"
    local new_folder="$2"

    # Extract the filename from the original path
    local filename=$(basename "$original_path")

    # Combine the new folder with the filename
    local new_path="$new_folder/$filename"

    echo "$new_path"
}

VOLUMES+=("-v" "$HOME/.pcpt/config:${CONFIG_PATH_OVERRIDE}")
VOLUMES+=("-v" "$HOME/.pcpt/log:${LOG_PATH_OVERRIDE}")

# Command-specific positional path rules
set_positional_path_rules() {
    local command=$1
    case "$command" in
        analyze|components|user-experience|domain-model|use-cases|business-logic|code-review|sequence)
            echo "last"
            ;;
        question|follow-up-question|extract-psm|generate-code)
            echo "second-last"
            ;;
        run-custom-prompt)
            echo "second-last last"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Get the first argument (command) and store it
COMMAND="$1"
shift  # Shift to process the remaining arguments

# Ensure the command is added to ARGS
ARGS+=("$COMMAND")

# Determine the path rules for the command
POS_PATH_RULES=$(set_positional_path_rules "$COMMAND")

# echo "$COMMAND -> $POS_PATH_RULES"

# Track if we're expecting a path after certain flags
PATH_TYPE=""

# List of switches that expect paths as the next argument
SWITCHES_EXPECT_PATH=("-o" "--output" "--domain-hints" "--spa-domain-hints" "--microservice-domain-hints" "--input-file" "--input-file2" "--domain" "--filter")

# List of switches that expect a non-path value (not a filesystem path)
SWITCHES_EXPECT_VALUE=("--mode" "-m" "--index" "--total" "--plantuml-server" "--plantuml-timeout" "--plantuml-retries")

# Process the remaining arguments and handle path mapping
POS_ARGS=()  # Collect positional arguments separately
EXPECTING_VALUE_FOR=""

while [[ "$#" -gt 0 ]]; do
    # First, if a previous switch expects a non-path value, consume it here
    if [[ -n "$EXPECTING_VALUE_FOR" ]]; then
        case "$EXPECTING_VALUE_FOR" in
            "--mode"|"-m")
                # Validate allowed values
                if [[ "$1" != "multi" && "$1" != "single" ]]; then
                    echo "Error: --mode must be 'multi' or 'single'. Got: '$1'"
                    exit 1
                fi
                # Pass the value straight through to container args
                ARGS+=("$1")
                EXPECTING_VALUE_FOR=""
                shift
                continue
                ;;
            "--plantuml-server")
                PLANTUML_SERVER_ENV="$1"
                EXPECTING_VALUE_FOR=""
                shift
                continue
                ;;
            "--plantuml-timeout")
                # basic numeric validation (allow int/float)
                if ! [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
                    echo "Error: --plantuml-timeout must be a number (seconds). Got: '$1'"
                    exit 1
                fi
                PLANTUML_TIMEOUT_ENV="$1"
                EXPECTING_VALUE_FOR=""
                shift
                continue
                ;;
            "--plantuml-retries")
                if ! [[ "$1" =~ ^[0-9]+$ ]]; then
                    echo "Error: --plantuml-retries must be an integer. Got: '$1'"
                    exit 1
                fi
                PLANTUML_RETRIES_ENV="$1"
                EXPECTING_VALUE_FOR=""
                shift
                continue
                ;;
            *)
                # Fallback: just pass through the value
                ARGS+=("$1")
                EXPECTING_VALUE_FOR=""
                shift
                continue
                ;;
        esac
    fi
    # Support equals-style for mode: --mode=multi or -m=single
    if [[ "$1" == --mode=* || "$1" == -m=* ]]; then
        MODE_VALUE="${1#*=}"
        if [[ "$MODE_VALUE" != "multi" && "$MODE_VALUE" != "single" ]]; then
            echo "Error: --mode must be 'multi' or 'single'. Got: '${MODE_VALUE}'"
            exit 1
        fi
        # Normalize to long form switch + value as separate args
        ARGS+=("--mode" "$MODE_VALUE")
        shift
        continue
    fi

    # Support equals-style for index: --index=3
    if [[ "$1" == --index=* ]]; then
        INDEX_VALUE="${1#*=}"
        # basic integer validation
        if ! [[ "$INDEX_VALUE" =~ ^[0-9]+$ ]]; then
            echo "Error: --index must be an integer. Got: '${INDEX_VALUE}'"
            exit 1
        fi
        ARGS+=("--index" "$INDEX_VALUE")
        shift
        continue
    fi
    # Support equals-style for total: --total=12
    if [[ "$1" == --total=* ]]; then
        TOTAL_VALUE="${1#*=}"
        if ! [[ "$TOTAL_VALUE" =~ ^[0-9]+$ ]]; then
            echo "Error: --total must be an integer. Got: '${TOTAL_VALUE}'"
            exit 1
        fi
        ARGS+=("--total" "$TOTAL_VALUE")
        shift
        continue
    fi
    # Support equals-style for PlantUML settings
    if [[ "$1" == --plantuml-server=* ]]; then
        PLANTUML_SERVER_ENV="${1#*=}"
        shift
        continue
    fi
    if [[ "$1" == --plantuml-timeout=* ]]; then
        VALUE="${1#*=}"
        if ! [[ "$VALUE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            echo "Error: --plantuml-timeout must be a number (seconds). Got: '${VALUE}'"
            exit 1
        fi
        PLANTUML_TIMEOUT_ENV="$VALUE"
        shift
        continue
    fi
    if [[ "$1" == --plantuml-retries=* ]]; then
        VALUE="${1#*=}"
        if ! [[ "$VALUE" =~ ^[0-9]+$ ]]; then
            echo "Error: --plantuml-retries must be an integer. Got: '${VALUE}'"
            exit 1
        fi
        PLANTUML_RETRIES_ENV="$VALUE"
        shift
        continue
    fi

    # Normalize --image boolean flag (supports --image and --image=true/false)
    if [[ "$1" == "--image" ]]; then
        ARGS+=("--image")
        shift
        continue
    fi
    if [[ "$1" == --image=* ]]; then
        IMAGE_VALUE="${1#*=}"
        case "${IMAGE_VALUE,,}" in
            1|true|yes|on)
                ARGS+=("--image")
                ;;
            0|false|no|off)
                # do not pass the flag through
                ;;
            *)
                echo "Error: --image expects true/false (or 1/0, yes/no, on/off). Got: '${IMAGE_VALUE}'"
                exit 1
                ;;
        esac
        shift
        continue
    fi

    #echo "Processing argument: $1"
    if [[ "$1" == -* ]]; then
        # It's a switch, check if it expects a path
        ARGS+=("$1")
        if [[ "$1" == "-o" || "$1" == "--output" ]]; then
            OUTPUT_PROVIDED=true  # Set the flag if output is provided
        fi
        # If this switch expects a non-path value, mark that we need to consume the next token
        if [[ " ${SWITCHES_EXPECT_VALUE[@]} " =~ " $1 " ]]; then
            EXPECTING_VALUE_FOR="$1"
            PATH_TYPE=""
            # Do not treat the next token as positional or a path; it will be consumed at the top of the loop
        elif [[ " ${SWITCHES_EXPECT_PATH[@]} " =~ " $1 " ]]; then
            PATH_TYPE="$1"
        else
            PATH_TYPE=""
        fi
    elif [[ -n "$PATH_TYPE" ]]; then
        # This argument is expected to be a path because it follows a switch that expects a path

        if [[ -f "$1" ]]; then
            FOLDER_PATH=$(dirname "$1")
            #echo "$1 is a file, using its directory: $FOLDER_PATH"
        else
            FOLDER_PATH="$1/.."  # If $1 is already a directory, use it as is
            #echo "$1 is a directory, using it as is: $FOLDER_PATH"
        fi

        case "$PATH_TYPE" in
            "-o" | "--output")
                map_path_to_volume "$FOLDER_PATH" "/output"
                # Pass only the leaf name into the container as /output/<leaf>
                ARGS+=("$(replace_folder_in_path "$1" "/output")")
                # On the host, PCPT_OUTPUT should be the parent directory, not include the leaf again
                PARENT_DIR="$(dirname "$1")"
                if [[ "$1" = /* ]]; then
                    # Absolute path
                    if [[ "$PARENT_DIR" == "." ]]; then
                        PCPT_OUTPUT="$(pwd)"
                    else
                        PCPT_OUTPUT="$PARENT_DIR"
                    fi
                else
                    # Relative path
                    if [[ "$PARENT_DIR" == "." ]]; then
                        PCPT_OUTPUT="$CURRENT_PATH_OVERRIDE"
                    else
                        PCPT_OUTPUT="$CURRENT_PATH_OVERRIDE/$PARENT_DIR"
                    fi
                fi
                ;;
            "--domain-hints")
                # Check if ~/.pcpt/hints/$1 exists as a file
                if [[ -f "$HOME/.pcpt/hints/$1" ]]; then
                    #echo "Found hint file: $HOME/.pcpt/hints/$1"  # If it exists as a file, do this
                    map_path_to_volume "$HOME/.pcpt/hints" "/app/hints"
                    ARGS+=("/app/hints/$1")
                else
                
                    # If it does not exist as a file, do something else
                    #echo "Hint file not found: $HOME/.pcpt/hints/$1"
                    ARGS+=("$(replace_folder_in_path "$1" "/app/hints")")
                fi
                ;;
            "--spa-domain-hints")
                # Check if ~/.pcpt/hints/$1 exists as a file
                if [[ -f "$HOME/.pcpt/hints/$1" ]]; then
                    # If it exists as a file, do this
                    map_path_to_volume "$HOME/.pcpt/hints" "/app/hints"
                    ARGS+=("/app/hints/$1")
                else
                    # If it does not exist as a file, do something else
                    ARGS+=("$(replace_folder_in_path "$1" "/app/hints")")
                fi
                ;;
            "--microservice-domain-hints")
                 # Check if ~/.pcpt/hints/$1 exists as a file
                if [[ -f "$HOME/.pcpt/hints/$1" ]]; then
                    # If it exists as a file, do this
                    map_path_to_volume "$HOME/.pcpt/hints" "/app/hints"
                    ARGS+=("/app/hints/$1")
                else
                    # If it does not exist as a file, do something else
                    ARGS+=("$(replace_folder_in_path "$1" "/app/hints")")
                fi
                ;;
            "--input-file")
                map_path_to_volume "$FOLDER_PATH" "/input"
                ARGS+=("$(replace_folder_in_path "$1" "/input")")
                ;;
            "--input-file2")
                map_path_to_volume "$FOLDER_PATH" "/input"
                ARGS+=("$(replace_folder_in_path "$1" "/input")")
                ;;
            "--domain")
                map_path_to_volume "$FOLDER_PATH" "/input"
                ARGS+=("$(replace_folder_in_path "$1" "/input")")
                ;;
            "--filter")
                # Check if ~/.pcpt/filters/$1 exists as a file
                if [[ -f "$HOME/.pcpt/filters/$1" ]]; then
                    # If it exists as a file, do this
                    map_path_to_volume "$HOME/.pcpt/filters" "/app/filters"
                    ARGS+=("/app/filters/$1")
                else
                    # If it does not exist as a file, do something else
                    ARGS+=("$(replace_folder_in_path "$1" "/app/filters")")
                fi
                ;;
            *)
                echo "Error: Unsupported PATH_TYPE '$PATH_TYPE'."
                exit 1
                ;;
        esac

        PATH_TYPE=""
    else
        # This is a positional argument
        POS_ARGS+=("$1")
    fi
    shift
done

# Inherit host env vars if not set via CLI flags
if [[ -z "$PLANTUML_SERVER_ENV" && -n "$PLANTUML_SERVER" ]]; then
    PLANTUML_SERVER_ENV="$PLANTUML_SERVER"
fi
if [[ -z "$PLANTUML_TIMEOUT_ENV" && -n "$PLANTUML_TIMEOUT" ]]; then
    PLANTUML_TIMEOUT_ENV="$PLANTUML_TIMEOUT"
fi
if [[ -z "$PLANTUML_RETRIES_ENV" && -n "$PLANTUML_RETRIES" ]]; then
    PLANTUML_RETRIES_ENV="$PLANTUML_RETRIES"
fi

# Prepare env var passthroughs for podman
if [[ -n "$PLANTUML_SERVER_ENV" ]]; then
    ENV_VARS+=("-e" "PLANTUML_SERVER=${PLANTUML_SERVER_ENV}")
fi
if [[ -n "$PLANTUML_TIMEOUT_ENV" ]]; then
    ENV_VARS+=("-e" "PLANTUML_TIMEOUT=${PLANTUML_TIMEOUT_ENV}")
fi
if [[ -n "$PLANTUML_RETRIES_ENV" ]]; then
    ENV_VARS+=("-e" "PLANTUML_RETRIES=${PLANTUML_RETRIES_ENV}")
fi

# Handle positional arguments based on the rules for the command
POS_COUNT=${#POS_ARGS[@]}
if [[ -n "$POS_PATH_RULES" ]]; then
    # Apply second last rule first, then last rule to avoid conflicts
    for RULE in $POS_PATH_RULES; do
        if [[ "$RULE" == "second-last" && "$POS_COUNT" -gt 1 ]]; then
            # Treat the second last positional argument as a path
            SECOND_LAST_ARG=${POS_ARGS[$((POS_COUNT-2))]}
            if [[ -f "$SECOND_LAST_ARG" || -d "$SECOND_LAST_ARG" ]]; then
                case "$COMMAND" in
                    "question"|"follow-up-question"|"extract-psm"|"run-custom-prompt")
                        map_path_to_volume "$SECOND_LAST_ARG" "/source_path"
                        ARGS+=("/source_path")
                        ;;
                    "generate-code")
                        map_path_to_volume "$SECOND_LAST_ARG" "/target_path"
                        ARGS+=("/target_path")
                        ;;
                esac
            else
                echo "Error: $SECOND_LAST_ARG is not a valid file or directory."
                exit 1
            fi
            # Keep the last argument for further processing
            POS_ARGS=("${POS_ARGS[@]:0:$((POS_COUNT-2))}" "${POS_ARGS[$((POS_COUNT-1))]}")
            POS_COUNT=${#POS_ARGS[@]}  # Recalculate after removing
        fi
        if [[ "$RULE" == "last" && "$POS_COUNT" -gt 0 ]]; then
            # Treat the last positional argument as a path
            LAST_ARG=${POS_ARGS[$((POS_COUNT-1))]}
            
            FOLDER_PATH=$(dirname "$LAST_ARG")

            case "$COMMAND" in
                "run-custom-prompt")
                    # Check if ~/.pcpt/prompts/$LAST_ARG exists as a file
                    if [[ -f "$HOME/.pcpt/prompts/$LAST_ARG" ]]; then
                        # If it exists as a file, do this
                        map_path_to_volume "$HOME/.pcpt/prompts/$FOLDER_PATH" "/app/prompts/custom"
                        ARGS+=("/app/prompts/custom/$LAST_ARG")
                    else
                        # If it does not exist as a file, do something else
                        ARGS+=("$(replace_folder_in_path "$LAST_ARG" "/app/prompts/custom")")
                    fi
                    ;;
                *)
                    # Fast sanity check: require the host path to exist (file OR directory)
                    SRC_ABS="$(realpath "$LAST_ARG" 2>/dev/null)" || SRC_ABS=""
                    if [[ -z "$SRC_ABS" || ! -e "$SRC_ABS" ]]; then
                        echo "Error: Source path not found on host: $LAST_ARG"
                        exit 2
                    fi
                    map_path_to_volume "$LAST_ARG" "/source_path"
                    ARGS+=("/source_path")
                    PCPT_SOURCE_PATH="$CURRENT_PATH_OVERRIDE/$LAST_ARG"
                    ;;
            esac
            POS_ARGS=("${POS_ARGS[@]:0:$((POS_COUNT-1))}")  # Remove the last positional argument
            POS_COUNT=${#POS_ARGS[@]}  # Recalculate after removing
        fi
    done
fi

# Add any remaining positional arguments that are not paths
ARGS+=("${POS_ARGS[@]}")

# Map current directory to /extvol/out if no output path is provided
if [[ "$OUTPUT_PROVIDED" == false ]]; then
    map_path_to_volume "$CURRENT_DIR" "/output"
    PCPT_OUTPUT=$CURRENT_DIR
fi

# If no command is given, run the container image with -h
if [[ -z "$COMMAND" ]]; then
    podman run --rm -i -t \
    ${VOLUMES[@]} \
    ${ENV_VARS[@]} \
    -e PCPT_CONFIG_PATH="$CONFIG_PATH_OVERRIDE" \
    greghodgkinson/pcpt:edge -h
    exit 0
fi

# Detect AWS profile and region from ~/.aws/config
AWS_CONFIG_FILE="$HOME/.aws/config"
PROFILE_FROM_CONFIG=""
REGION_FROM_CONFIG=""

if [[ -f "$AWS_CONFIG_FILE" ]]; then
    PROFILE_FROM_CONFIG=$(awk '/^\[profile / {gsub(/^\[profile /,""); gsub(/\]/,""); print; exit}' "$AWS_CONFIG_FILE")
    if [[ -z "$PROFILE_FROM_CONFIG" ]]; then
        PROFILE_FROM_CONFIG="default"
    fi
    REGION_FROM_CONFIG=$(awk -v profile="$PROFILE_FROM_CONFIG" '
        $0 ~ "\\[profile "profile"\\]" {found=1; next}
        /^\[profile / {found=0}
        found && $1 == "region" {print $3; exit}
    ' "$AWS_CONFIG_FILE")
fi

if [[ -z "$PROFILE_FROM_CONFIG" ]]; then
    PROFILE_FROM_CONFIG="default"
fi

if [[ -z "$REGION_FROM_CONFIG" ]]; then
    REGION_FROM_CONFIG="us-east-2"
fi

CONTAINER_AWS_PROFILE="$PROFILE_FROM_CONFIG"
CONTAINER_AWS_REGION="$REGION_FROM_CONFIG"

# Build the podman command with the volume mappings
PODMAN_CMD="podman run --privileged --rm -i -t \
  ${VOLUMES[@]} \
  -e PCPT_PROGRAM_PATH=\"${PROGRAM_PATH_OVERRIDE}\" \
  -e PCPT_CONFIG_PATH=\"$CONFIG_PATH_OVERRIDE\" \
  -e PCPT_PROMPTS_PATH=\"$PROMPTS_PATH_OVERRIDE\" \
  -e PCPT_HINTS_PATH=\"$HINTS_PATH_OVERRIDE\" \
  -e PCPT_LOG_PATH=\"$LOG_PATH_OVERRIDE\" \
  -e PCPT_FILTERS_PATH=\"$FILTERS_PATH_OVERRIDE\" \
  -e PCPT_SOURCE_PATH=\"$PCPT_SOURCE_PATH\" \
  -e PCPT_OUTPUT=\"$PCPT_OUTPUT\" \
  greghodgkinson/pcpt:edge \"${ARGS[@]}\""

 #echo "Volumes mapped: ${VOLUMES[@]}"
 #echo "Args: ${ARGS[@]}"

# Output the command before running it
#echo "Running command: $PODMAN_CMD"

# Run the Docker container with all necessary volume mappings
podman run --privileged --rm -i -t \
  ${VOLUMES[@]} \
  -v "$HOME/.aws:/root/.aws:rw" \
  -e AWS_PROFILE="$CONTAINER_AWS_PROFILE" \
  -e AWS_REGION="$CONTAINER_AWS_REGION" \
  -e AWS_SDK_LOAD_CONFIG=1 \
  -e AWS_EC2_METADATA_DISABLED=true \
  ${ENV_VARS[@]} \
  -e PCPT_PROGRAM_PATH="${PROGRAM_PATH_OVERRIDE}" \
  -e PCPT_CONFIG_PATH="$CONFIG_PATH_OVERRIDE" \
  -e PCPT_PROMPTS_PATH="$PROMPTS_PATH_OVERRIDE" \
  -e PCPT_HINTS_PATH="$HINTS_PATH_OVERRIDE" \
  -e PCPT_LOG_PATH="$LOG_PATH_OVERRIDE" \
  -e PCPT_FILTERS_PATH="$FILTERS_PATH_OVERRIDE" \
  -e PCPT_SOURCE_PATH="$PCPT_SOURCE_PATH" \
  -e PCPT_OUTPUT="$PCPT_OUTPUT" \
  greghodgkinson/pcpt:edge "${ARGS[@]}"
