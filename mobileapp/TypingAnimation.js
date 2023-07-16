import React, { useEffect, useState } from 'react';
import { Text } from 'react-native';
import * as Animatable from 'react-native-animatable';

const TypingAnimation = ({ text, fontSize, animationSpeed }) => {
  const [displayText, setDisplayText] = useState('');

  useEffect(() => {
    let display = '';
    text.split('').forEach((char, index) => {
      setTimeout(() => {
        display += char;
        setDisplayText(display);
      }, index * animationSpeed); // the typing speed, adjust accordingly
    });
  }, [text]);

  return (
    <Animatable.Text animation="bounceIn" style={{ color: '#000', fontSize: fontSize }}>
      {displayText}
    </Animatable.Text>
  );
};

export default TypingAnimation;
