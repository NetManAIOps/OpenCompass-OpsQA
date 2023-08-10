workdir := /mnt/mfs/opsgpt/opencompass

clear_outputs:
	python $(workdir)/tools/clear_outputs.py $(workdir)/outputs/default
	python $(workdir)/tools/clear_outputs.py $(workdir)/outputs/demo
	python $(workdir)/tools/clear_outputs.py $(workdir)/outputs/opsqa
