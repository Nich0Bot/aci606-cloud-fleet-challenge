# Target architecture

```text
                    multiple virtual robots
                  R01   R02   R03   R04 ...
                    \    |     |    /
                     \   |     |   /
                      v  v     v  v
                  +-------------------+
                  |  Cloud API        |
                  |  POST /telemetry  |
                  |  POST /predict    |
                  +---------+---------+
                            |
              +-------------+-------------+
              |                           |
              v                           v
       persistent storage             ML artifact
        telemetry/history            model.joblib
              |                           |
              +-------------+-------------+
                            |
                            v
                  +-------------------+
                  | Dashboard/Metrics |
                  +-------------------+
```

The starter implements only the top-left part in memory. Exercise 4 is to complete the system.
