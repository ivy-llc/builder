#!/bin/bash -e
docker run --rm -it -v "$(pwd)":/ivy_builder ivydl/ivy-builder:latest python3 -m pytest ivy_builder_tests/
