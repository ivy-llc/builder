#!/bin/bash

# Functions #
# ----------#

# create json files for specification loading from their default files, replacing BASE_DIRs with the mount directories
create_json_files() {
  for json_fname in "$1"/*.json.defaults; do
    if [ -f "$json_fname" ]; then
      # shellcheck disable=SC2001
      new_json_fname="$(echo "${json_fname}" |sed -e "s|.defaults||g")";
      if [ -f "new_json_fname" ]; then
        rm "new_json_fname"
      fi
      cp "$json_fname" "$new_json_fname"
      sed -i "s|BASE_DATASET_DIR|$2|g" "$new_json_fname"
      sed -i "s|BASE_LOG_DIR|$3|g" "$new_json_fname"
    fi
  done
  for json_fname in "$1"/*.json.example; do
      # shellcheck disable=SC2001
    new_json_fname="$(echo "${json_fname}" |sed -e "s|.example||g")";
    if [ -f "$json_fname" ] && ! [ -f "$new_json_fname" ]; then
      cp "$json_fname" "$new_json_fname"
      sed -i "s|BASE_DATASET_DIR|$2|g" "$new_json_fname"
      sed -i "s|BASE_LOG_DIR|$3|g" "$new_json_fname"
    fi
  done
}

# recursively create json files in all sub-directories
recursive_update() {
  # apply this method recursively to all sub-directories
  for file in "$1"/*; do
    if [ -d "${file}" ] ; then
      recursive_update "${file}" "$2" "$3"
    fi
  done
  # copy new json files for current directory
  create_json_files "$1" "$2" "$3"
}

# end
end () {
  echo >&2 "$@"
  exit 1
}

# Run #
# ----#

# initialise option flag with a false value
OPT_R='false'

# process all options supplied on the command line
while getopts ':r' 'OPTKEY'; do
    case ${OPTKEY} in
        'r')
            # Update the value of the option x flag we defined above
            OPT_R='true'
            ;;
        '?')
            echo "INVALID OPTION -- ${OPTARG}" >&2
            exit 1
            ;;
        ':')
            echo "MISSING ARGUMENT for option -- ${OPTARG}" >&2
            exit 1
            ;;
        *)
            echo "UNIMPLEMENTED OPTION -- ${OPTKEY}" >&2
            exit 1
            ;;
    esac
done

# remove all options processed by getopts.
shift $(( OPTIND - 1 ))
[[ "${1}" == "--" ]] && shift

# verify dataset and log mount directories are specified
if [ $# -eq 0 ]
  then
    end "dataset and log mount directories must be specified"
fi

# change to a sub-directory if specified
if [ $# -eq 3 ]
  then
    cd "$3" || exit
fi

create_json_files "$PWD" "$1" "$2"

# call the same script in all sub-directories if -r is specified
if ${OPT_R}; then
  recursive_update "$PWD" "$1" "$2"
fi