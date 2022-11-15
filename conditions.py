"""
Author: Cody
Date Created: 11/15/2022
Last Modified: 11/15/2022
Purpose: 
    Code for the Condions Class. Allows for the easy creation and maintaining of condition information.
"""

class Conditions:
    def __init__(self):
        self.conditionName = ""
        self.description = ""
        self.links = []
    
    def generateResults(self, model_outcome):
        if model_outcome == 0:
            self.conditionName = "Benign"
            self.description = "Benign skin lesions are a form of non-cancerous skin growth. Benign skin lessions is a classification used for any number of conditions ranging from common acne to rashes or sores"
            self.links = [('Cleveland Clinic', "https://my.clevelandclinic.org/health/diseases/12014-moles-freckles-skin-tags-lentigines-and-seborrheic-keratoses#:~:text=What%20are%20benign%20skin%20lesions,benign%20lentigines%2C%20and%20seborrheic%20keratoses"), ('John Hopkins Medicine', "https://www.hopkinsmedicine.org/health/conditions-and-diseases/other-benign-skin-growths")]
        elif model_outcome == 1:
            self.conditionName = "Malignant"
            self.description = "Malignant skin lesions are those were the cells are growing out of control and spreading into other, unrelated tissues. This is the type of condition that people commonly refer to as cancer."
            self.links = [('WebMD', "https://www.webmd.com/melanoma-skin-cancer/ss/skin-cancer-and-skin-lesions-overview"), ('CDC', "https://www.cdc.gov/cancer/skin/basic_info/index.htm")]