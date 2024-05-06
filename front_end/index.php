<?php include_once "database.php";?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    <link rel="stylesheet" href="css/owl.carousel.min.css">  
    <link rel="stylesheet" href="css/owl.theme.default.min.css">
    <link rel="stylesheet" href="css/template.css" >
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
</head>

<?php
$max_items_per_page=10;
$page=isset($_GET['p'])?$_GET['p']:1;
$offset = ($page-1)*$max_items_per_page;
$sql_result = "SELECT * FROM `item_list` LIMIT $offset, $max_items_per_page";
$result = $pdo->query($sql_result);
$sql_total_itmes = "SELECT COUNT(*) AS total FROM item_list";
$total_itmes = $pdo->query($sql_total_itmes)->fetchColumn();
$total_pages = ceil($total_itmes/$max_items_per_page);
?>

<body>
    <main>
        <section>
            <div class="container">
                <div class = "row" >
                    <div class = "col-lg-12 mg-5 pb-3">
                        <h1 calss>Gorgeous Nepenthes Series</h1>
                    </div>
                    <div class="owl-carousel owl-theme">
                         <div class="owl-carousel-info-wrap item">
                              <img src="images/veitchii_1_resized.jpg" class="owl-carousel-image img-fluid" alt="">
                         </div>
                         <div class="owl-carousel-info-wrap item">
                              <img src="images/truncata_1_resized.jpg" class="owl-carousel-image img-fluid" alt="">
                         </div>
                         <div class="owl-carousel-info-wrap item">
                              <img src="images/ventricosa_1_resized.jpg" class="owl-carousel-image img-fluid" alt="">
                         </div>
                         <div class="owl-carousel-info-wrap item">
                              <img src="images/veitchii_2_resized.jpg" class="owl-carousel-image img-fluid" alt="">
                         </div>
                         <div class="owl-carousel-info-wrap item">
                              <img src="images/truncata_2_resized.jpg" class="owl-carousel-image img-fluid" alt="">
                         </div>  
                         <div class="owl-carousel-info-wrap item">
                              <img src="images/ventricosa_2_resized.jpg" class="owl-carousel-image img-fluid" alt="">
                         </div>                                                                                                                            
                    </div>
                </div>
            </div>
        </section>
        <section class="latest-podcast-section section-padding pb-0" id="section_2">
            <div class="container">
                <div class="row ">
                    <div class="col-sm-2 col-md-2" >
                        <div class="tag-container" role="group" aria-label="Basic example" id="categories">

                            <h2>Species</h2>
                            <div class="row ">
                                <button class="tag all selected">All</button>
                                <button class="tag" id="truncata">truncata</button>
                                <button class="tag" id="veitchii">veitchii</button>
                                <button class="tag" id="ventricosa">ventricosa</button>             
                            </div>
                        </div>
                    </div>
                    <div  class="col-sm-10 col-md-10">
                        <div class="row" id="items-container">
                        </div>
                    </div>  
                </div>  
            </div> 
        </section>  
                        <?php
                        if ($result->rowCount() > 0) {
                            while ($row = $result->fetch(PDO::FETCH_ASSOC)) {
                        ?>
                        <div class="block">
                            <h3><?php echo $row["species"]; ?></h3>
                            <p><?php echo $row["serial"]; ?></p>
                        </div>
                        <?php
                            }
                        }
                        ?>
        <div>
        <?php
        if ($page > 1) {
            echo "<a href='index.php?page=".($page - 1)."'>Previous</a> ";
        }
        for ($i = 1; $i <= $total_pages; $i++) {
            echo "<a href='index.php?page=$i'>$i</a> ";
        }
        if ($page < $total_pages) {
            echo "<a href='index.php?page=".($page + 1)."'>Next</a>";
        }
        ?>
        </div>
    </main>
    <footer>
    </footer>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script> 
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.7.1/jquery.min.js" integrity="sha512-qFOQ9YFAeGj1gDOuUD61g3D+tLDv3u1ECYWqT82WQoaWrOhAY+5mRMTTVsQdWutbA5FORCnkEPEgU0OF8IzGvA==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script src="js/jquery.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js" integrity="sha384-YvpcrYf0tY3lHB60NNkmXc5s9fDVZLESaAA55NDzOxhy9GkcIdslK1eN7N6jIeHz" crossorigin="anonymous"></script> 
    <script src="https://cdnjs.cloudflare.com/ajax/libs/OwlCarousel2/2.3.4/owl.carousel.min.js" integrity="sha512-bPs7Ae6pVvhOSiIcyUClR7/q2OAsRiovw4vAkX+zJbw3ShAeeqezq50RIIcIURq7Oa20rW2n2q+fyXBNcU9lrw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>     
    <script src="js/bootstrap.bundle.min.js"></script>
    <script src="js/owl.carousel.min.js"></script>
    <script src="js/custom.js"></script>
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.5.2/js/bootstrap.min.js"></script>
</body>
</html>