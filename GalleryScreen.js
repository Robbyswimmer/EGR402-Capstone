import React, { Component } from 'react';
import { StyleSheet, Text, View, Image, ScrollView } from 'react-native';
import TopBar from '../../components/TopBar';


import axios from 'axios';
import Constants from 'expo-constants';
import { Collapse, CollapseHeader, CollapseBody } from "accordion-collapse-react-native";
import { Thumbnail, List, ListItem, Separator, Row, Right } from 'native-base';
import { greaterThan } from 'react-native-reanimated';

function getPicCover(pic){
  if(typeof pic.cover !== "object"){
      return pic.cover
  }
  else return pic.cover.url.replace('t_thumb', 't_1080p_2x')
}

class PicsList extends Component {
  state = {
    pics: []
  }
  componentDidMount = async ()=>{
    var pics = await this.props.retrievepics();
    this.setState({ pics: pics });
  }
  render(){
    const pics = this.state.pics;
    if(pics.length){
      console.log("pics:", pics[0].checksum);
    }

    const title = this.props.title;
    return (
      <ScrollView>
      <View >
      <Collapse isExpanded={true}>
        <CollapseHeader>
          <Text style={styles.tab}>{title}</Text>
        </CollapseHeader>
        <CollapseBody >
          {pics.map((pics)=>{
            return (
              <ListItem key={pics.id} style={styles.picstile}>
                <Image
                  source={{ uri:`https:${getpicsCover(pics)}` }}
                  style={[styles.photo, { width: 150, height: 190 }]}
                />
                <View style = {[styles.rightContainer]}>

                  
                </View>
              </ListItem>
            )
          })}
        </CollapseBody>
      </Collapse>
      </View>
      </ScrollView>
    )    
  }
}


export default function LibraryScreen({ navigation }) {

  return (
    <View style={{backgroundColor:""}}>
      <View style={styles.container} >
        <TopBar handleProfilePress={handleProfilePress} handleHomePress={handleHomePress} />
      </View>
      <View>
      </View>
      <PicsList/>
      <PicsList/>
    </View>
  )

  function handleHomePress() {
    navigation.navigate('Home');
  }

}

const styles = StyleSheet.create({
  
  container: {
    flex: 1,
    marginTop: Constants.statusBarHeight,
  },
  title: {
    backgroundColor:'#F06795',
    marginTop: Constants.statusBarHeight,
    fontSize: 20,
    color: "azure",
    width: "auto",
  },

  tab: {
    marginTop: Constants.statusBarHeight,
    height:"auto",
    width:"auto",
    alignItems: 'center',
    justifyContent: 'space-evenly',
    backgroundColor: '#F06795',
    fontSize: 25,
    color: "azure",
    textAlign:'center',

  },
  picstile:{
    height:200, 
    width:"auto", 
    backgroundColor: 'aliceblue',
    alignItems:'center',
    borderColor: "darkslateblue"
  },
  photo: {
    position: "absolute",
    justifyContent: 'center',
    alignItems: 'center',
    height: '100%',
    resizeMode: 'cover',
    borderRadius: 20,
  },
  textPrimary: {
    marginTop: 60,
    color: 'black',
    fontSize: 20,
    fontWeight: 'bold',
    fontFamily: 'sans-serif-light'
  },
  textSecondary:{
    color: 'black',
    fontSize: 15,
    // fontWeight: 'bold',
    fontFamily: 'sans-serif-light'
  },
  rightContainer: {
    marginLeft: 175,
    marginBottom: 100,
    textAlign: 'center'
  }

})