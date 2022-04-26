import * as React from 'react';
import { StatusBar, FlatList, Image, Animated, Text, View, Dimensions, StyleSheet, TouchableOpacity } from 'react-native';
const { width, height } = Dimensions.get('screen');
import {Asset} from 'expo-asset'

const camURI = Asset.fromModule(require('../cam_os/images/camera.jpg')).uri
const galleryURI = Asset.fromModule(require('../cam_os/images/gallery.jpg')).uri
const settingsURI = Asset.fromModule(require('../cam_os/images/settings.jpg')).uri

// const camera = "Camera"
// const gallery = "Gallery"
// const settings = "Settings"

const data = [
  camURI,
  galleryURI,
  settingsURI,
    
];

// const stringData = [
//   camera,
//   gallery,
//   settings
// ]

const imageW = width * 0.7;
const imageH = imageW * 1.54;

export default () => {
  const scrollX = React.useRef(new Animated.Value(0)).current;
  return(
        <View style={{ flex: 1, backgroundColor: '#000' }}>
            <StatusBar hidden/>

            <View 
              style={StyleSheet.absoluteFillObject}
            >
              {data.map((image, index) => {
                const inputRange = [
                  (index - 1) * width,
                  index * width,
                  (index + 1) * width
                ]
                const opacity = scrollX.interpolate({
                  inputRange,
                  outputRange: [0, 1, 0]
                })
                return <Animated.Image
                  key={`image-${index}`}
                  source={{uri: image}}
                  style={[
                    StyleSheet.absoluteFillObject,
                    {
                      opacity
                    }
                  ]}
                  blurRadius={50}
                />
              })}
            </View>

            <Animated.FlatList
            onScroll={Animated.event(
              [{nativeEvent: {contentOffset: {x: scrollX} }}],
              {useNativeDriver: true}
            )}
              data = {data}
              keyExtractor={(_, index) => index.toString()}
              horizontal
              pagingEnabled
              renderItem={({item}) => {
                return <View style={
                  {width, justifyContent: 'center', alignItems: 'center', 
                  shadowColor: '#000',
                  shadowOpacity: .5,
                  shadowOffset: {
                    width: 0,
                    height: 0,
                  },
                  shadowRadius: 20
                
                }}>
                  <Image source={{uri: item}} style={{
                    width: 300,//imageW,
                    height: 400, //imageH,
                    resizeMode: 'contain',
                    borderRadius: 100
                  }}/>
                </View>
              }}
            />
          

        </View>
    );
};