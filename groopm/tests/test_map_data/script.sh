#!/bin/bash
NAME=$0
short_description() {
  echo "Generate a set of mapping test sequences"
}
usage() {
  echo "Usage: $NAME CONTIGS_FILE CONTIG_HITS [CONTIG_HITS...]"
}
help() {
  short_description; usage
}
improper_usage() {
  usage >&2; exit 1
}
# Globals
CONTIGS_FILE=
# Utilities
info() {
  if test "$LOG_INFO"; then echo $@ >&2; fi
}
warn() {
  echo $@ >&2
}
error() {
  echo "$NAME:" $@ >&2; exit 1
}
get_sequence_names() {
  grep '>' |sed 's/^>//'
}
get_unique_hits() {
  cat $@ |cut -f1 |sort |uniq
}
filter_sequence_names() {
  local file filter_list tmp_join_file
  file=$1
  filter_list=$2
  tmp_join_file=$(basename $file)'.tmp.filtered'
  join -t '' -o 1.1 $file $filter_list > $tmp_join_file
  sort -m $file $tmp_join_file |uniq -u
  rm $tmp_join_file
}
select_sequence_names() {
  local tmp_hits_file tmp_file
  tmp_file=$(basename $CONTIGS_FILE)'.tmp.names'
  tmp_hits_file=$tmp_file'.tmp.hits'
  get_sequence_names < $CONTIGS_FILE |sort > $tmp_file
  get_unique_hits $@ > $tmp_hits_file
  filter_sequence_names $tmp_file $tmp_hits_file |shuf -n100
  rm $tmp_file $tmp_hits_file
}
get_sequences() {
  local contig_file list
  contigs_file=$1
  list=$2
  echo $list >&2
  #awk -v 'begin { while (1) { getline <"'$list'"; if ($0) { array[">" $0]=1 } else break } }; /^>/ { $name=arrray[$0]; if ($name) print }; if ($name) { print }' $contigs_file
  awk 'begin { while (1) { getline seq <"'$list'"; print "hi"; if ($seq) { array[">" $seq]=1 } else { print $ERRNO; break } }; for (val in array) print $val }' $contigs_file
}
# Main
test $# -ge 2 || improper_usage
for file; do
  test -f $file || error "$file: Not a file"
done
CONTIGS_FILE=$1
shift
tmp_select_names=$(basename $CONTIGS_FILE)'.tmp.select'
select_sequence_names $@ > $tmp_select_names
get_sequences $CONTIGS_FILE $tmp_select_names