// This script runs on the arduino nano and provides a bridge between the computer and the ZIB-2 converting a digitial signal to TTL signal for placing annotations
const int triggerPin = 3;
int incomingByte;
const int triggerTime = 30; //in ms

void setup() {
  // initialize serial:
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(triggerPin, OUTPUT);
}

void loop() {
  // see if there's incoming serial data:
  if (Serial.available() > 0) {
    // read the oldest byte in the serial buffer:
    incomingByte = Serial.read();
    // if it's anything other than 0, send trugger abd turn on board LED:
    if (incomingByte != 0) {
      digitalWrite(triggerPin, HIGH);
      digitalWrite(LED_BUILTIN, HIGH);
      delay(100);
      digitalWrite(triggerPin, LOW);
      digitalWrite(LED_BUILTIN, LOW);
      incomingByte = 0;
    }
  }
}
