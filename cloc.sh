#!/bin/sh
cloc --exclude-dir=.cache,.pytest_cache,build,KAS.egg-info .
