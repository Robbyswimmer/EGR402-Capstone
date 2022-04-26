import * as React from 'react';
import { View, Image, Text, StyleSheet } from 'react-native';
import Constants from 'expo-constants';
import TopBar from '../../components/TopBar';
import { withOrientation } from 'react-navigation';


export default function settingsScreen({navigation}) { 


     function handleHomePress() {
       navigation.navigate('Home');
     }

    return (
    <Background>

    <View style = {styles.container2}> 
        <View style = {styles.topBar}>
            <TopBar handleProfilePress = {handleProfilePress} handleHomePress={handleHomePress} />
        </View>

        
        <View style ={styles.container}>

        <Image source = {require ('../../images/Userpro.jpg')}
        style = {{width: 130, height: 150}}/>

        <Text style= {styles.textPrimary}>{'\n'}{'\n'}Exposure</Text>

        <Text style= {styles.textPrimary}>{'\n'}{'\n'}Saturation</Text>

        <Text style= {styles.textPrimary}>{'\n'}{'\n'}Balance</Text>

        <Text style= {styles.textPrimary}>{'\n'}{'\n'}Tone</Text>

        <Text style= {styles.textPrimary}>{'\n'}{'\n'}Levels</Text>
        </View>
        
    </View>
    </Background>
    

    )

    }


const styles = StyleSheet.create({
    // container2: {
    //     flex: 1,
    //     marginTop: '10%',
    //     width: '145%',
    // },
    container: {
       height: '100%',
       backgroundColor: theme.colors.surface,
       alignItems:'center',
       fontSize: 32, 
       paddingTop: 90,
       marginTop: Constants.statusBarHeight,
       color: theme.colors.primary,
    },

    topBar: {
        flex: 1,
        marginTop: '30%',
    },

    photo: {
        height: 100,
        borderRadius: 20,
    },

    textPrimary: {
        color: theme.colors.primary,
    }
 })