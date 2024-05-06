(function ($) {
    "use strict";
    $('.owl-carousel').owlCarousel({
        center: true,
        loop: true,
        margin: 30,
        autoplay: true,
        responsiveClass: true,
        responsive:{
            0:{
                items: 2,
            },
            767:{
                items: 3,
            },
            1200:{
                items: 4,
            }
        }
    });
})(window.jQuery);

$(document).ready(function(){
    console.log("Document ready");
    fetchItems([]);
    $('.tag').click(function(){
        console.log("Button clicked");
        $(this).toggleClass('selected');
        if($(this).hasClass('all')) {
            $('.tag').removeClass('selected');
            $(this).toggleClass('selected');
            fetchItems([]);
            return;
        }
        $('.all').removeClass('selected');
        var selected_tags = $('.tag.selected').map(function() {
            return this.id;
        }).get();
        console.log("Selected tags: ", selected_tags);
        fetchItems(selected_tags);
    });
    function getSpeciesNameFromFileName(fileName) {
        var normalFileName = fileName.replace(/%20/g, ' ');
        var parts = normalFileName.split("'");
        if (parts.length >= 2) {

            var speciesName = parts[1];
            var prefix = parts[0];
            var fullFileName = prefix + "'" + speciesName + parts.slice(2).join("'");
            return {speciesName: speciesName, fullFileName: fullFileName};
        } else {
            return null; 
        }
    }
    function fetchItems(selected_tags) {
        if (selected_tags.length === 0) {
            selected_tags = [];
        }
        $.ajax({
            url: 'backend.php',
            method: 'POST',
            data: {selected_tags: selected_tags},
            dataType: 'json',
            success: function(response){
                $('#items-container').empty();
                $.each(response, function(index, item) {
                    var species = item.species;
                    var serial = item.serial;
                    var source = item.source;              
                    var imgSrc = '../../imgs/' + encodeURIComponent(species + '_' + serial + '_' + source) + '.jpg';
                    var block = $('<div class="custom-block custom-block-full col-sm-6 col-md-6">');
                    console.log("Block:", block);
                    var speciesElem = $('<h3>').text(item.species);
                    var serialElem = $('<p>').text(item.serial);
                    var sourceElem = $('<p>').text(item.source);
                    var imgElem = $('<img>').attr('src', imgSrc).addClass('block-image');
                    block.append(imgElem);
                    block.append(speciesElem);
                    block.append(serialElem);
                    block.append(sourceElem);    
                    $('#items-container').append(block);
                });
            },
            error: function(xhr, status, error) {
                console.error("AJAX error:", error);
            }
        });
    }
});


