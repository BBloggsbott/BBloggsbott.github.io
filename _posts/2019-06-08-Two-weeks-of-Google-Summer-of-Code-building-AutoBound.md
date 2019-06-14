It has been almost two weeks since coding officially started with Google Summer of code. As mentioned in my previous [post](https://medium.com/@bbloggsbott/google-summer-of-code-autobound-863860006fc0), I have been selected to work as a Student Developer for OpenStreetMap through Google Summer of Code.

I have been working on my project and the experience so far has been great. In this post, I'll be explaining what I've done and you can find the code in the [development](https://github.com/BBloggsbott/autobound/tree/development/) branch of this [repo](https://github.com/BBloggsbott/autobound/).

## The Action Class

First, I had to create an Action class. The methods of this class will be called when an action occurs. For my plugin, I wanted the user to be able to select an area in the map view and this area had to be processed. For this functionality, I had to extend the MapMode class (makes the Action a map mode) and implement the SelectionEnded Interface and override the necessary functions.

Once this was done, I had to add the action class to the MapMode menu in JOSM. I did this by overriding the mapFrameInitialized method of the Plugin class. This method is called when the map frame has been initialized. In this method, I added the action class as an Icon Toggle Button to the map modes menu.

## Utilities

Next, I created classes that can provide utilities for the working of the plugin. The following classes were created:

- DataUtils
- NetworkUtils
- MapUtils

These classes provide functions that can prepare data to send to the server, communicate with the server, add points (nodes) from a DataSet on the map view.

## Tests

Once all this was done, I started writing unit tests to make sure that the classes are doing what I expect them to do. For this, I used JUnit and JOSM's test utilities.

## Data Collection

The heart of AutoBound is the Deep Learning model that will recognize buildings from satellite images. To do this, the model must understand what buildings look like in satellite images. For the model to understand that, it is going to need a lot of images (generally, the more the better).

To collect data, I added another component to AutoBound. This can be found under the Tools menu of JOSM (is the plugin is enabled). When this is used, it collects the images of buildings and sends data to the server for some processing.

### Getting data from JOSM

When "Collect Data" is used, it sends some data from JOSM to the server. This is what happens (I use East North coordinate system as Latitude - Longitude will be a bit complex as the curvature of the earth is not uniform):

- It gets the DataSet from the currently active Data Layer
- From this DataSet, all the **Way**s with the tag **building** are extracted.
- For each building extracted from the previous step: 
  - Extract the image of the building with some padding. (The padding is half the length of the building along that axis on either side)
  - Net, we collect the nodes that constitute the building (in the right order)
  - Collect some metadata of the building like Id, timestamp, minimum north and east, maximum north and east.
  - All this data is bundled up into a JSON object and sent to the server (The satellite image is base64 encoded).

When the server gets this data, it begins to process it as described as below:

- The meta data is stored in a csv file
- The image is decoded and saved
- A new blank image of the same size as the image from the previous step is created and painted black.
- Using the data of the nodes sent by the plugin, the new image is painted (white) such that we get a image of the building and it is saved.

Below are sample images from the data collection process

![Satellite Image](https://cdn-images-1.medium.com/max/800/1*hk3Mok_a_taHO-t5rTKoZg.png)

*Satellite Image. Source: [OpenAerialMap](https://openaerialmap.org/). Provider: [Oyibo VFX](https://map.openaerialmap.org/#/4.643714500000007,50.89814351061963,18/user/5cf2cf9f8e78c70006c96201/5cf51ef4c25f7e00059bac6b?_k=0a13it)*

![Segmented Image](https://cdn-images-1.medium.com/max/800/1*93jgVYrVcLxE0YE7josXuw.png)

*Segmented image generated using data from [OpenStreetMap](https://www.openstreetmap.org/). Click [here](https://www.openstreetmap.org/way/129017941) to view the way.*

## What I've learned

Students rarely get a chance to build applications using just docs of dependencies. This project was my first time building something using docs alone and it is taught to be how to use docs efficiently and how to write good documentation.

Data collection is another area where a lot of people lack experience in. This project has helped me with that too.

The main aim of writing this article was to show my progress and also help someone who might have an idea for a new plugin for JOSM get started. I think I've accomplished that.

Below are the links to my repos (development branch):

* [AutoBound](https://github.com/BBloggsbott/autobound/tree/development)
* [AutoBound Server](https://github.com/BBloggsbott/autoboundserver/tree/development)

*Note: All the data provided by OpenStreetMap is available under the Open Database License.*