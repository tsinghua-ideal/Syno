#!/bin/sh
cloc --exclude-dir=.cache,.pytest_cache,.vscode,build,KAS.egg-info,save .
