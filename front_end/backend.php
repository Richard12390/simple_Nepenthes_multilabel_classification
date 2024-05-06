<?php
$servername = "localhost";
$username = "root";
$password = "";
$dbname = "nepenthes_multilabel_classification";

try {
    $conn = new PDO("mysql:host=$servername;dbname=$dbname", $username, $password);
    $conn->setAttribute(PDO::ATTR_ERRMODE, PDO::ERRMODE_EXCEPTION);
} catch(PDOException $e) {
    http_response_code(500);
    echo json_encode(["error" => "Database connection failed: " . $e->getMessage()]);
    exit(); 
}

if(isset($_POST['selected_tags'])) {
    $selected_tags = $_POST['selected_tags'];
    $items = array();
    $sql = "SELECT species, serial, source FROM `item_list`";
    if (!empty($selected_tags)) {
        $sql .= " WHERE ";
        foreach ($selected_tags as $index => $tag) {
            if ($index > 0) {
                $sql .= " AND ";
            }
            $sql .= "`$tag` = 1";
        }
    }
    try {
        $stmt = $conn->prepare($sql);
        $stmt->execute();
        while ($row = $stmt->fetch(PDO::FETCH_ASSOC)) {
            $items[] = $row;
        }

        header('Content-Type: application/json');
        echo json_encode($items);
    } catch(PDOException $e) {
        http_response_code(500);
        echo json_encode(["error" => "SQL query failed: " . $e->getMessage()]);
    }
} else {
    try {
        $stmt = $conn->query("SELECT species, serial, source FROM `item_list`");
        $items = $stmt->fetchAll(PDO::FETCH_ASSOC);
        header('Content-Type: application/json');
        echo json_encode($items);
    } catch(PDOException $e) {
        http_response_code(500);
        echo json_encode(["error" => "SQL query failed: " . $e->getMessage()]);
    }
}
?>
