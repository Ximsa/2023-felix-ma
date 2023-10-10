#!/bin/bash                       
FOO="["
FOO+=$(find . -name result.json | xargs cat | tr "\n" "," | head -c -1)
FOO+="]"
echo $FOO | jq > summary.json
