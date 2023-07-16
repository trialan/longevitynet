import React from 'react';
import { Text } from 'react-native';
import TypingAnimation from './TypingAnimation';

export default class Result extends React.Component {
  render() {
    return (
      <Text>Your predicted life expectancy is {this.props.lifeExpectancy} years.</Text>
    );
  }
}

