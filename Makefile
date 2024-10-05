# Set your default jupyter port here. 
# Avoid 8888 if you run local notebooks on that!
PORT=8888
# Your app root in the container.
APP_DIR=/ws
# The docker image to launch.  *** SET TO YOUR image:version! ***
APP_TAG=ws
.SILENT:

##
## Launch a short python script fromm our container, mapping two folders
##    Local          Container       Notes
##    -----------------------------------------------------
##    ./Data      -> /Data           Put data here!
##    ./Results   -> /Results        Write results here
##    ./Notebooks -> /Notebooks      Find notebooks here!
##    -----------------------------------------------------
python: ##
	docker run \
	-v $(PWD)/Notebooks/\:$(APP_DIR)/Notebooks/ \
	-v $(PWD)/Data\:$(APP_DIR)/Data \
	-v $(PWD)/Results\:$(APP_DIR)/Results \
	$(APP_TAG) \
	python mess.py

##
## Launch jupyter notebook from our container, mapping two folders
##    Local          Container       Notes
##    -----------------------------------------------------
##    ./Data      -> /Data           Put data here
##    ./Results   -> /Results        Write results here
##    ./Notebooks -> /Notebooks      Find notebooks here
##    -----------------------------------------------------
jupyter: ##
	docker run \
	-p $(PORT)\:$(PORT) \
	-v $(PWD)/Notebooks/\:$(APP_DIR)/Notebooks/ \
	-v $(PWD)/Data\:$(APP_DIR)/Data \
	-v $(PWD)/Results\:$(APP_DIR)/Results \
	$(APP_TAG) \
	jupyter notebook --ip 0.0.0.0 --port $(PORT) --no-browser --allow-root 

prune: ##
	docker image prune
	docker container prune
