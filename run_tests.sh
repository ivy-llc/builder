#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/builder unifyai/ivy-builder:latest python3 -m pytest ivy_builder_tests/
