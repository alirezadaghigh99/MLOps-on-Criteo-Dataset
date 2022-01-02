set -u

echo Running local inference...

EXAMPLES_FILE=$1
SCHEMA_FILE=$2

python `dirname "$(readlink -f "$0")"`/client.py \
  --num_examples 3 \
  --examples_file ${EXAMPLES_FILE} \
  --schema_file ${SCHEMA_FILE} \
  --server 127.0.0.1:9000
