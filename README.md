# Travelling-Postman-Problems-Algorithms-
This repository contains a series of different TSP algorithms which will simulate the optimal path for an agent positioned in a middle of a square grid of X points

# Details to run TSPP app
Go into the TSPP/myapp directory and run
```bash
python3 manage.py runserver
```

# This is the file structure

# This what to expect from the user interface
![alt text](/TSPP/website.png "User Interface")

```mermaid
flowchart TB;
       LiDAR & StereoCamera -->|Point Cloud 2D|J["Sensor Fusion (vis)"];

       IMU & GPS --> B["Sensor Fusion (odom)"];

       USER -->|Boundaries|H[Global Planning];

       J -->|Vecs of Objects rel. to ROMIE?| I;

       B -->|"Odometry (real vel.)"| C[Control];

       H -->|Waypoints| I;

       I[Local Planing] -->|"reference heading (or ref. vel.)"| C;       

       
       C --->|"velocity output"| X[Motor Controllers];
       
       F[Payload] --->|Completion Status| C;

       C -->|Drill Command| F;

       E[Motor Encoders] --> B;

```

# This is the react web interface to choose the sampling point coordinate in Africa
![alt text](/TSPP/website_gif.gif "User Interface")
