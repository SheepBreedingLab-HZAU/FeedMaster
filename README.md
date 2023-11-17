
# LAD-RCNN

![](https://img.shields.io/static/v1?label=python&message=3.8&color=blue)
![](https://img.shields.io/static/v1?label=TensorFlow&message=2.8&color=<COLOR>)
![](https://img.shields.io/static/v1?label=Numpy&message=1.21&color=<COLOR>)
![](https://img.shields.io/static/v1?label=license&message=MIT&color=green)

This is official implementation of "[Feed Formula Optimization Based on Backpropagation Algorithm]()"


# Framework
![The overall pipeline of the FeedMaster](https://github.com/SheepBreedingLab-HZAU/FeedMaster/blob/main/Figure/Flowchart.jpg)


#  Paper Your Data

 - The data is passed into the program in the form of a JSON file.
 - After placing the specified JSON file in the "files" folder, run Main.py to calculate the feed formula
 - The conf file contains sections such as [Standard], [Resource_n], and [Initial Feed Formula].
 - [Standard] includes three parts: StandTitle, Standard, and StandWeight

``` 
 - StandTitle is the name of the nutritional components in the feeding standards.
 - You can enter any number of nutrients, separated by commas between them	
 
 - Standard is the content of each nutritional components in the feeding standard.
 - The quantity must be consistent with StandTitle, separated by commas.

 - StandWeight is the weight of each nutritional component in the feeding standard, with a default value of 1. 
 - The quantity must be consistent with StandTitle, separated by commas
```

 - The nutritional composition, price, and usage restrictions of feed ingredients are entered in [Resource_n], where n is a variable. Please specify different n for different feed ingredients, otherwise it may cause the program to fail to execute.

``` 
 -  Each [Resource_n] section contains four items: Name, Price, Nutrition Content, and Usage Limit
 -  The quantity of numbers entered in the Nutrition Content must be consistent with the StandTitle
```

 - [Initial Feed Formula] is an optional parameter. If there is no input, the program will give a default value

# 2 Start optimizing feed formulation

```
 python Main.py

If you have any questions, please contact Ling Sun,E-mail:ling.sun-01@qq.com
```
 
# Requirement
- [ ] TensorFlow  2.8.0
- [ ] Python > 3.7
- [ ] Numpy > 1.20


