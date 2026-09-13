\# UAV Engine Fault-to-Action Strategy



\## Purpose



The fault classifier identifies the most likely engine fault from

telemetry features. The operator strategy converts that diagnosis

into a practical response.



The strategy is advisory and intended for the Digital Twin prototype.

It does not replace certified flight-control or safety systems.



\---



\## 1. Healthy / None



\### Detection

Normal engine telemetry remains within expected operating ranges.



\### Key indicators

\- Stable RPM

\- Stable EGT and CHT

\- Normal oil pressure and oil temperature

\- Normal vibration

\- Health score remains high



\### Recommended operator action

\- Continue normal operation.

\- Continue monitoring telemetry.

\- No immediate maintenance action required.



\---



\## 2. Misfire



\### Detection

Intermittent combustion-related behaviour produces instability in

engine telemetry.



\### Key indicators

\- RPM variability

\- EGT variability

\- Increased vibration variability

\- Intermittent fault occurrences



\### Recommended operator action

\- Reduce engine load if operationally possible.

\- Monitor RPM, EGT and vibration closely.

\- Inspect combustion/ignition-related components after operation.

\- If instability becomes severe, initiate the appropriate

&#x20; operational shutdown/return procedure.



\---



\## 3. Overheat



\### Detection

Increasing engine temperature behaviour, particularly in EGT and CHT.



\### Key indicators

\- EGT increase

\- CHT increase

\- Increasing temperature-related health degradation

\- Higher thermal variability



\### Recommended operator action

\- Reduce engine load/RPM where operationally possible.

\- Monitor EGT and CHT continuously.

\- Inspect the thermal/cooling system.

\- If temperature continues to rise toward critical limits,

&#x20; initiate the appropriate protective operational procedure.



\---



\## 4. Cooling Degradation



\### Detection

Progressive increase in CHT with comparatively smaller changes

in other engine parameters.



\### Key indicators

\- CHT increase

\- CHT range/slope increase

\- EGT comparatively less affected than in overheat

\- Gradual health-score degradation



\### Recommended operator action

\- Reduce engine load if possible.

\- Closely monitor CHT.

\- Inspect cooling-system performance.

\- Check for restricted airflow, cooling-system degradation or

&#x20; related thermal-management problems during maintenance.



\---



\## 5. Oil Issue



\### Detection

Lubrication-related deterioration characterized primarily by

decreasing oil pressure and increasing oil temperature.



\### Key indicators

\- Oil pressure decrease

\- Oil temperature increase

\- Oil-related health-score degradation



\### Recommended operator action

\- Reduce engine load immediately where operationally possible.

\- Monitor oil pressure and oil temperature.

\- If pressure continues to fall or temperature rises excessively,

&#x20; initiate the appropriate protective operational procedure.

\- Inspect oil level, lubrication system and related components.



\---



\## 6. Sensor Drift



\### Detection

Telemetry from a sensor exhibits bias, gradual drift or a stuck

value.



\### Key indicators

\- Unusual sensor trend

\- Persistent bias

\- Abnormally low temporal variation for a stuck sensor

\- Sensor behaviour inconsistent with related engine parameters



\### Recommended operator action

\- Treat the affected sensor reading with reduced confidence.

\- Compare the affected measurement against correlated sensors.

\- Trigger sensor recalibration when appropriate.

\- Replace the sensor if recalibration does not restore reliable

&#x20; measurements.

\- Avoid making critical decisions from an isolated drifting sensor.



\---



\## 7. Vibration Fault



\### Detection

Increasing vibration level and variability associated with a

mechanical/rotating-component abnormality.



\### Key indicators

\- Increased vibration mean

\- Increased vibration standard deviation

\- Increased vibration range

\- Increasing vibration trend



\### Recommended operator action

\- Reduce engine RPM/load where operationally possible.

\- Monitor vibration continuously.

\- Inspect rotating/mechanical components.

\- If vibration continues increasing toward an unsafe condition,

&#x20; initiate the appropriate protective operational procedure.



\---



\# Operator Decision Logic



The Digital Twin can convert the classifier output into an advisory

severity/action level.



| Condition | Advisory level | Suggested response |

|---|---|---|

| None + high health score | NORMAL | Continue monitoring |

| Fault detected + mild health degradation | CAUTION | Monitor closely / reduce load if appropriate |

| Fault detected + significant health degradation | WARNING | Reduce load and investigate |

| Severe fault indicators + rapidly degrading health | CRITICAL | Initiate appropriate protective operational procedure |



\## Important design principle



The fault classification and health score should be considered

together.



The classifier answers:



> "What fault is most likely occurring?"



The health score answers:



> "How healthy is the engine's current operating condition?"



This allows the Digital Twin to provide both diagnosis and

interpretable condition assessment.

