<?php
$servername = "localhost";
$username = "prtissot";
$password = "database";

// Create connection
$conn = new mysqli($servername, $username, $password, "Yelp");

// Check connection
if ($conn->connect_error) {
  die("Connection failed: " . $conn->connect_error);
}
echo "Connected successfully\n";


function map_name($conn, $city){
    $handle = fopen($city.'_review.csv', "r");
    if ($handle) {
        while (($line = fgetscsv($handle)) !== false) {
            var_dump($line);

        }

        fclose($handle);
    }
}

$cities = array ('Philadelphia','Tampa', 'Indianapolis', 'New Orleans', 'Tucson');


foreach($cities as $city){
    map_name($conn, $city);
}

$conn->close();
?>