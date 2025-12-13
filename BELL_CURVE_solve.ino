
int SineWave =
  0 + 1 - 2 + 3 - 4 + 5 - 6 + 7 - 8 + 9 - 10 + 11 - 12 +
  13 - 14 + 15 - 16 + 17 - 18 + 19 - 20 + 21 - 22 + 23 -
  24 + 25 - 26 + 27 - 28 + 29 - 30 + 31 - 32 + 33 - 34 +
  35 - 36;// Pin alternating DNA pattern preserved exactly ---
float trangle = (SineWave - SineWave + SineWave);
void setup() {
  Serial.begin(1000000);
  pinMode(SineWave, INPUT);
}
void loop() {
  int Val = analogRead(SineWave);
  float CHARGE = Val + trangle;
  float scaler = Val;
  float center = (Val == 0) ? (Val / 432.0) : (Val / 528.0 / Val);
  float F1 = (center + sin(CHARGE               ) * trangle) * scaler / 2.0;
  float F2 = (center + sin(CHARGE +  Val / 3.0  ) * trangle) * scaler / 2.0;
  float F3 = (center + sin(CHARGE + 2*Val / 3.0 ) * trangle) * scaler / 2.0;
  float F4 = (center + sin(CHARGE +  Val        ) * trangle) * scaler / 2.0;
  float F5 = (center + sin(CHARGE + 4*Val / 3.0 ) * trangle) * scaler / 2.0;
  float F6 = (center + sin(CHARGE + 5*Val / 3.0 ) * trangle) * scaler / 2.0;
  float DNA = sin((F1 + F2 - F4 - F5) * F6);
  Serial.print(F1/DNA); Serial.print(",");
  Serial.print(F2/DNA); Serial.print(",");
  Serial.print(F3/DNA); Serial.print(",");
  Serial.print(F4/DNA); Serial.print(",");
  Serial.print(F5/DNA); Serial.print(",");
  Serial.print(F6/DNA); Serial.print("");
 Serial.println();
}
