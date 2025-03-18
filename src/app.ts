import { NumberHandler } from './NumberHandler';
import * as readline from 'readline';

const numberHandler = new NumberHandler();
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

console.log("Welcome to the Next Random Number Identifier!");
console.log("Enter numbers one by one and I'll tell you how they relate to the previous one.");
console.log("Type 'exit' to quit or 'reset' to clear the last number.");

function promptForNumber() {
  rl.question('Enter a number: ', (input) => {
    if (input.toLowerCase() === 'exit') {
      console.log('Goodbye!');
      rl.close();
      return;
    }

    if (input.toLowerCase() === 'reset') {
      numberHandler.resetLastNumber();
      console.log('Last number has been reset.');
      promptForNumber();
      return;
    }

    const number = parseFloat(input);
    if (isNaN(number)) {
      console.log('Please enter a valid number or "exit" to quit.');
    } else {
      console.log(numberHandler.identifyNextNumber(number));
    }
    
    promptForNumber();
  });
}

promptForNumber();
