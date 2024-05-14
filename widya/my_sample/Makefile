OUTNAME_RELEASE = my_sample
OUTNAME_DEBUG   = my_sample_debug
EXTRA_DIRECTORIES = ../common
SAMPLE_DIR_NAME = $(shell basename $(dir $(abspath $(firstword $(MAKEFILE_LIST)))))
MAKEFILE ?= ../Makefile.config
include $(MAKEFILE)
