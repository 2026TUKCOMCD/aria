#include "Relay.h"
#include "Arduino.h"
#include <avr/eeprom.h>

byte Relay::count = 0;
byte Relay::globalState = 0;

//Public Methods

Relay::Relay(byte relayPin, int eepromByte){
	id = count++;
	pin = relayPin;
	this->eepromByte = eepromByte;
	pinMode(pin,OUTPUT);
	globalState = eeprom_read_byte(&this->eepromByte);
	digitalWrite(pin,state = (bitRead(globalState,id)));
}

byte Relay::setState(byte newState){
	if((newState||0)==state && ((newState&&switchableToOn) || (!newState&&switchableToOff))){
		digitalWrite(pin, state = !newState);
		saveState();
	}
	return !state;
}

byte Relay::switchState(){
	if((state&&switchableToOn) || (!state&&switchableToOff)){
		digitalWrite(pin,state = !state);
		saveState();
	}
	return !state;
}

/*byte Relay::switchState(){
	if(!stateLocked) {
		digitalWrite(pin,state = !state);
		saveState();
	}
	return !state;
}*/

byte Relay::getState(){
	return !state;
}

void Relay::echoState(Stream *serialPort){
	if (getState()) serialPort->write("RelayON#");
	else serialPort->write("RelayOFF#");
}

void Relay::setSwitchableToOn(bool switchableToOn){
	this->switchableToOn = switchableToOn;
}

void Relay::setSwitchableToOff(bool switchableToOff){
	this->switchableToOff = switchableToOff;
}

bool Relay::isSwitchableToOn(){
	return switchableToOn;
}

bool Relay::isSwitchableToOff(){
	return switchableToOff;
}
/*
void Relay::lockState(){
	stateLocked = true;
}

void Relay::unlockState(){
	stateLocked = false;
}

bool Relay::isStateLocked(){
	return stateLocked;
}*/

//Private Methods

void Relay::saveState(){
	bitWrite(globalState, id, state);
	eeprom_write_byte(&this->eepromByte,globalState);
}