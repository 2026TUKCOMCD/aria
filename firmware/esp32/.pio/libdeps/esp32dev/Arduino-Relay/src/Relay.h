#ifndef Relay_h
#define Relay_h

#define ON 1
#define OFF 0

#include "Arduino.h"

class Relay {

public:
	Relay(byte relayPin, int eepromByte = 1023);
	byte setState(byte newState);
	byte switchState();
	byte getState();
	void echoState(Stream *serialPort);
	void setSwitchableToOn(bool switchableToOn);
	void setSwitchableToOff(bool switchableToOff);
	bool isSwitchableToOn();
	bool isSwitchableToOff();

private:
	static byte count;
	static byte globalState;
	byte id;
	byte pin;
	byte state;
	uint8_t eepromByte;
	void saveState();
	bool switchableToOn = true;
	bool switchableToOff = true;
};

#endif
