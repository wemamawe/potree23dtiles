#  potree to cesium 3dtiles
## Thanks to Potree and cesium for their contributions for rendering large point clouds
>> https://github.com/potree  
>> https://github.com/potree/PotreeConverter  
>> https://github.com/AnalyticalGraphicsInc/3d-tiles/tree/master/specification



# useage:
## potreeConvert
```
# generate compressed LAZ files instead of the default BIN format.
./PotreeConverter.exe C:/data.las -o C:/potree_converted --output-format LAS -p pageName
```
## potree23dtiles
The code is only for testing purposes,please see 'testConvert' function for specific usage.  
you should change these values as yours:
>> bbox、src、dst、proj_param

## cesium
```
    var tileset = new Cesium.Cesium3DTileset({ url: "http://127.0.0.1/test/tileset.json" });
   
    tileset.readyPromise.then(function(data) {
        viewer.scene.primitives.add(data);
    }
    
```