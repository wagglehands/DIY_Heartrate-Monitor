
/* ECG Analyzer ino file
Written by : Manivannan 
E-mail: manivannan0212@gmail.com 
Project: https://www.hackster.io/manivannan/ecg-analyzer-powered-by-edge-impulse-24a6c2
*/

/* Includes ---------------------------------------------------------------- */
#include <ecg_analyzer_inference.h>
#include <Arduino_LSM9DS1.h>


extern "C" {
#include "ecg_analyzer.h"
}
#define EI_CLASSIFIER_SENSOR 1






/*************************************OLED*********************************/
#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 // OLED display width, in pixels
#define SCREEN_HEIGHT 64 // OLED display height, in pixels

// Declaration for SSD1306 display connected using software SPI (default case):
#define OLED_MOSI   11
#define OLED_CLK   13
#define OLED_DC    9
#define OLED_CS    10
#define OLED_RESET 8
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT,
                         OLED_MOSI, OLED_CLK, OLED_DC, OLED_RESET, OLED_CS);

/* Comment out above, uncomment this block to use hardware SPI
  #define OLED_DC     6
  #define OLED_CS     7
  #define OLED_RESET  8
  Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT,
  &SPI, OLED_DC, OLED_RESET, OLED_CS);
*/

#define NUMFLAKES     10 // Number of snowflakes in the animation example

#define LOGO_HEIGHT   16
#define LOGO_WIDTH    16
static const unsigned char PROGMEM logo_bmp[] =
{ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0xFE, 0x42, 0x42, 0x42, 0x42, 0x42, 0x02, 0x00, 0x00, 0x00, 0x00,
  0xFE, 0x02, 0x02, 0x02, 0x04, 0x04, 0x08, 0xF0, 0x00, 0x00, 0x00, 0x00, 0xF0, 0x0C, 0x04, 0x02,
  0x02, 0x02, 0x82, 0x84, 0x80, 0x00, 0x00, 0x00, 0x00, 0xFE, 0x42, 0x42, 0x42, 0x42, 0x42, 0x02,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xFC, 0x38,
  0x60, 0xC0, 0x80, 0x80, 0xC0, 0x30, 0x38, 0xFE, 0x00, 0x00, 0x00, 0x00, 0xFE, 0x82, 0x82, 0x82,
  0x82, 0x82, 0x84, 0x7C, 0x00, 0x00, 0x00, 0x00, 0xFE, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00, 0x00,
  0x0F, 0x08, 0x08, 0x08, 0x08, 0x0C, 0x06, 0x03, 0x00, 0x00, 0x00, 0x00, 0x03, 0x04, 0x08, 0x08,
  0x08, 0x08, 0x08, 0x08, 0x07, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x08, 0x08, 0x08, 0x08, 0x08, 0x08,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x00,
  0x00, 0x00, 0x03, 0x01, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x0C, 0x08, 0x08, 0x08, 0x08, 0x0C, 0x07,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
  0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00
};


/*********************************** Preidctions variable ***********************/
int NormalStateCounter = 0;
int AtrialFibrillationCounter = 0;
int First_DegreeHeartBlockCounter = 0;
int predictionCounter = 1;
const char *prediction;
float THRESHOLD_CYCLE_TIME = 60000;
bool OnecycleCheckEnable = 0;
float CycleCheckCounter = 0;

uint8_t NORMAL_STATE = 1;
uint8_t ATRIAL_FIBRILLATION_STATE = 2;
uint8_t FIRST_DEGREE_HEART_BLOCK_STATE = 3;

int MaximumOccuredEvent(int n1 , int n2, int n3);
uint8_t Event = 0;
float percentageCaln = 0;
/* Private variables ------------------------------------------------------- */
static bool debug_nn = false; // Set this to true to see e.g. features generated from the raw signal
static uint32_t run_inference_every_ms = 200;
static rtos::Thread inference_thread(osPriorityLow);
static float buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE] = { 0 };
static float inference_buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE];

/* Forward declaration */
void run_inference_background();

/**
  @brief      Arduino setup function
*/
void setup()
{
  // put your setup code here, to run once:
  Serial.begin(115200);

  // SSD1306_SWITCHCAPVCC = generate display voltage from 3.3V internally
  if (!display.begin(SSD1306_SWITCHCAPVCC)) {
    Serial.println(F("SSD1306 allocation failed"));
    for (;;); // Don't proceed, loop forever


  }




  // Clear the buffer
  display.clearDisplay();
  display.setTextSize(2); // Draw 2X-scale text
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10, 0);
  display.println(F("ECG"));
  display.display();      // Show initial text
  display.println(F("ANALYZER"));
  display.display();      // Show initial text
  delay(3000);
  // Clear the buffer
  display.clearDisplay();
  display.setTextSize(1); // Draw 2X-scale text
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(10, 0);
  display.println(F("Powered by"));
  display.display();      // Show initial text
  display.setTextSize(2); // Draw 2X-scale text
  display.println(F("Edge "));
  display.display();      // Show initial text
  display.println(F("   Impulse "));
  display.display();      // Show initial text
  //Serial.println("Edge Impulse Inferencing Demo");
  delay(3000);
  display.clearDisplay();
  display.setTextSize(1); // Draw 2X-scale text
  display.println(F("Analyzing..."));
  display.display();      // Show initial text

  //pinMode(10, INPUT); // Setup for leads off detection LO +
  //pinMode(11, INPUT); // Setup for leads off detection LO -
  ECG_Filter_initialize();
  inference_thread.start(mbed::callback(&run_inference_background));
}

/**
  @brief      Printf function uses vsnprintf and output using Arduino Serial

  @param[in]  format     Variable argument list
*/
void ei_printf(const char *format, ...) {
  static char print_buf[1024] = { 0 };

  va_list args;
  va_start(args, format);
  int r = vsnprintf(print_buf, sizeof(print_buf), format, args);
  va_end(args);

  if (r > 0) {
    Serial.write(print_buf);
  }
}

/**
   @brief      Run inferencing in the background.
*/
void run_inference_background()
{
  // wait until we have a full buffer
  delay((EI_CLASSIFIER_INTERVAL_MS * EI_CLASSIFIER_RAW_SAMPLE_COUNT) + 100);

  // This is a structure that smoothens the output result
  // With the default settings 70% of readings should be the same before classifying.
  ei_classifier_smooth_t smooth;
  ei_classifier_smooth_init(&smooth, 10 /* no. of readings */, 7 /* min. readings the same */, 0.8 /* min. confidence */, 0.3 /* max anomaly */);

  while (1) {
    // copy the buffer
    memcpy(inference_buffer, buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE * sizeof(float));

    // Turn the raw buffer in a signal which we can the classify
    signal_t signal;
    int err = numpy::signal_from_buffer(inference_buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, &signal);
    if (err != 0) {
      ei_printf("Failed to create signal from buffer (%d)\n", err);
      return;
    }

    // Run the classifier
    ei_impulse_result_t result = { 0 };

    err = run_classifier(&signal, &result, debug_nn);
    if (err != EI_IMPULSE_OK) {
      ei_printf("ERR: Failed to run classifier (%d)\n", err);
      return;
    }

    // print the predictions
    //  ei_printf("Predictions ");
    //ei_printf("(DSP: %d ms., Classification: %d ms., Anomaly: %d ms.)",
    //      result.timing.dsp, result.timing.classification, result.timing.anomaly);
    ei_printf(": ");

    // ei_classifier_smooth_update yields the predicted label
    const char *prediction = ei_classifier_smooth_update(&smooth, &result);
    ei_printf("%s ", prediction);



    if (strcmp(prediction, "uncertain") != 0)
    {
      predictionCounter++;

    }
    if (strcmp(prediction, "Normal") == 0)
    {
      NormalStateCounter++;
      Serial.print("NormalStateCounter: \t");
      Serial.println(NormalStateCounter);
    }
    else if (strcmp(prediction, "Atrial Fibrillation") == 0)
    {
      AtrialFibrillationCounter++;
      Serial.print("AtrialFibrillationCounter: \t");
      Serial.println(AtrialFibrillationCounter);

    }
    else if (strcmp(prediction, "First-Degree Heart Block") == 0)
    {

      First_DegreeHeartBlockCounter++;
      Serial.print("First_DegreeHeartBlockCounter: \t");
      Serial.println(First_DegreeHeartBlockCounter);

    }

    // print the cumulative results
    ei_printf(" [ ");
    for (size_t ix = 0; ix < smooth.count_size; ix++) {
      ei_printf("%u", smooth.count[ix]);
      if (ix != smooth.count_size + 1) {
        ei_printf(", ");
      }
      else {
        ei_printf(" ");
      }
    }
    ei_printf("]\n");

    delay(run_inference_every_ms);
  }

  ei_classifier_smooth_free(&smooth);
}

/**
  @brief      Get data and run inferencing

  @param[in]  debug  Get debug info if true
*/
void loop()
{
  while (1) {
    while (1) {

      if (OnecycleCheckEnable == 0)
      {

        CycleCheckCounter = millis();
        OnecycleCheckEnable = 1;

      }




      // Determine the next tick (and then sleep later)
      uint64_t next_tick = micros() + (EI_CLASSIFIER_INTERVAL_MS * 1000);

      // roll the buffer -3 points so we can overwrite the last one
      numpy::roll(buffer, EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE, -3);

      ecg_analyzer(analogRead(A0));
      /*
        Serial.print(calculated_ecg_values[0]);
        Serial.print("\t");
        Serial.print(calculated_ecg_values[1]);
        Serial.print("\t");
        Serial.println(calculated_ecg_values[2]);
      */


      buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 3] = calculated_ecg_values[0];
      buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 2] = calculated_ecg_values[1];
      buffer[EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE - 1] = calculated_ecg_values[2];

      // and wait for next tick
      uint64_t time_to_wait = next_tick - micros();
      delay((int)floor((float)time_to_wait / 1000.0f));
      delayMicroseconds(time_to_wait % 1000);


      if (millis() - CycleCheckCounter > THRESHOLD_CYCLE_TIME && (OnecycleCheckEnable == 1))
      {
        OnecycleCheckEnable = 0; //reset

        percentageCaln = ((NormalStateCounter * 100) / predictionCounter);
        display.clearDisplay();
        display.setTextSize(1); // Draw 2X-scale text
        display.setCursor(10, 0);
        display.print(F("Normal Rate: "));
        display.display();      // Show initial text
        display.setCursor(10, 10);
        display.setTextSize(3); // Draw 2X-scale text
        display.println(percentageCaln);
        display.display();      // Show initial text
        delay(3000);
        display.clearDisplay();

        percentageCaln = ((AtrialFibrillationCounter * 100) / predictionCounter);
        display.setCursor(10, 0);
        display.setTextSize(1); // Draw 2X-scale text
        display.print(F("AtrialFibrillation Rate : "));
        display.display();      // Show initial text
        display.setCursor(10, 25);
        display.setTextSize(3); // Draw 2X-scale text
        display.println(percentageCaln);
        display.display();      // Show initial text
        delay(3000);
        display.clearDisplay();
        percentageCaln = ((First_DegreeHeartBlockCounter * 100) / predictionCounter);
        display.setTextSize(1); // Draw 2X-scale text
        display.setCursor(10, 0);
        display.print(F("AV_Block 1: "));
        display.display();      // Show initial text
        display.setCursor(10, 12);
        display.setTextSize(3); // Draw 2X-scale text
        display.println(percentageCaln);
        display.display();      // Show initial text
        delay(3000);
        display.clearDisplay();






        Event = MaximumOccuredEvent(NormalStateCounter, AtrialFibrillationCounter, First_DegreeHeartBlockCounter);

        switch (Event)
        {

          case 1:
            {

              Serial.print(" OUTPUT : NORMAL_STATE \t");
              percentageCaln = ((NormalStateCounter * 100) / predictionCounter);
              Serial.println(percentageCaln);
              display.clearDisplay();
              display.setTextSize(1); // Draw 2X-scale text
              display.setCursor(10, 0);
              display.print(F("Normal Rate: "));
              display.display();      // Show initial text
              display.setCursor(10, 12);
              display.setTextSize(3); // Draw 2X-scale text
              display.println(percentageCaln);
              display.display();      // Show initial text
              break;
            }

          case 2:
            {
              Serial.print(" OUTPUT : ATRIAL_FIBRILLATION_STATE \t");
              percentageCaln = ((AtrialFibrillationCounter * 100) / predictionCounter);
              Serial.println(percentageCaln);
              display.clearDisplay();
              display.setTextSize(1); // Draw 2X-scale text
              display.setCursor(10, 0);
              display.print(F("AtrialFibrillation Rate: "));
              display.display();      // Show initial text
              display.setCursor(10, 25);
              display.setTextSize(3); // Draw 2X-scale text
              display.println(percentageCaln);
              display.display();      // Show initial text
              break;
            }


          case 3:
            {
              Serial.print(" OUTPUT : FIRST_DEGREE_HEART_BLOCK_STATE \t");
              percentageCaln = ((First_DegreeHeartBlockCounter * 100) / predictionCounter);
              Serial.println(percentageCaln);
              display.clearDisplay();
              display.setTextSize(1); // Draw 2X-scale text
              display.setCursor(10, 0);
              display.print(F("First_DegreeHeartBlock Rate: "));
              display.display();      // Show initial text
              display.setCursor(10, 12);
              display.setTextSize(3); // Draw 2X-scale text
              display.println(percentageCaln);
              display.display();      // Show initial text
              break;
            }


          default:
            {
              Serial.println("Uncertain");
              break;
            }

        }

        delay(1000);

        //Reset counters for next cycle check
        First_DegreeHeartBlockCounter = 0;
        AtrialFibrillationCounter = 0;
        NormalStateCounter = 0;
        predictionCounter = 0;


      }



    }
  }
}



int MaximumOccuredEvent(int n1 , int n2, int n3)
{
  int MaxEvent;

  if (n1 >= n2) {
    if (n1 >= n3)
    {
      MaxEvent = 1;
    }

    else
    {
      MaxEvent = 3;
    }
  } else {
    if (n2 >= n3)
    {
      MaxEvent = 2;
    }
    else
    {
      MaxEvent = 3;
    }
  }

  return MaxEvent;

}




void testdrawbitmap(void) {
  display.clearDisplay();

  display.drawBitmap(
    (display.width()  - LOGO_WIDTH ) / 2,
    (display.height() - LOGO_HEIGHT) / 2,
    logo_bmp, LOGO_WIDTH, LOGO_HEIGHT, 1);
  display.display();
  delay(1000);
}
