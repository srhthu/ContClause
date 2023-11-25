var contract_data = null;
var contract_id = 0; // update after get contract data
var pop_span_info = null; // object of the pop window

$(document).ready(function(){
    get_history();
    get_clauses();

    get_contract(-1) // Get the last one

    $("#btn_prev").click(on_click_prev);
    $("#btn_next").click(on_click_next);
    $("#btn_jump").click(on_click_jump);
    $("#show_space").click(create_contract);
    default_fs = parseInt($("span").css('font-size'));
    $("#font-size").val(default_fs);
    $("#font-size-display").text(default_fs);
    $("#font-size").change(on_change_font_size);
    
})

// API Calls

function get_history(){
    $.ajax({
        url:"/get-history",
        type:"GET",
        contentType: "application/json",
        success: callback_history,
        timeout:2000,
        async: false,
    })
}

function get_clauses(){
    $.ajax({
        url:"/get-clauses",
        type:"GET",
        contentType: "application/json",
        success: callback_clauses,
        timeout:2000,
        async: false
    })
}

function get_contract(index){
    $.ajax({
        url:"/get-contract",
        type:"POST",
        data: JSON.stringify({'index': index}),
        dataType: "json",
        contentType: "application/json",
        success: callback_contract,
        timeout:2000
    });
}

// API Callbacks

function callback_history(data){
    // Display browse history
    var his_dom = $("#history").empty();
    ul = $("<ul></ul>").attr('class', "list-group clause-list");
    for (var i=0; i<data.length; i++){
        var cont_idx = data[i].index;
        var cont_title = data[i].title;
        // var his_str = cont_idx + '&nbsp' + cont_title
        line = $("<a></a>")
                .append($("<span></span>")
                            .append($("<b></b>").text(cont_idx)))
                .append($("<span></span>").text(cont_title))
                .attr('data-index', cont_idx)
                .attr('class', 'list-group-item list-group-item-action')
                .attr('href', 'javascript:;')
                .click(function(){get_contract($(this).attr('data-index'))})
        ul.append(line);
    }
    his_dom.append(ul)
}

function callback_clauses(data){
    // Populate clause buttons
    var clause_dom = $("#clause_sel").empty();
    clause_dom.append(create_clause_list(data));
}

function callback_contract(data){
    // data has attributes 'title' and 'contract_text'
    contract_data = data;
    contract_id=data.idx;
    $("#ipt_index").val(data.idx);
    get_history()
    // disable selected clause type
    $("ul.clause-list>.list-group-item").removeClass('active')
    create_contract()
}

// Content Generations

function create_clause_list(data){
    ul = $("<ul></ul>").attr('class', "list-group clause-list");
    for (var i=0; i<data.length; i++){
        ul.append(create_clause_button(i, data[i]))
    }
    return ul
}

function create_clause_button(i, data){
    var category = data.category;
    var desc = data.desc;
    var group = data.group;
    btn = $("<span></span>").html(category)
        
        
    return $("<a></a>")
        .append($("<span></span>"))
        .append(btn)
        .attr('title', desc)
        .attr('class', 'list-group-item list-group-item-action')
        .attr('data-id', i)
        .attr('href', 'javascript:;')
        .click(on_click_clause_button)
        
}

function create_contract(){
    // contract_data is a dict of title, contract_text, atom_spans, impossible
    var data = contract_data;

    // show title
    $("#contract-title").text(data.title);

    // Disable impossible clause types
    $("#clause_sel .list-group-item").each(function(index, value){
        emoji=$(value).children("span:nth-child(1)");
        if (data.impossible[index]) {
            $(value).addClass('disabled');
            emoji.html("&#x274c");
        }
        else {
            $(value).removeClass('disabled');
            emoji.html("&#x2705");
        };
    })

    // Create atom spans
    var cont_dom = $("#contract").empty();
    // cont_dom.append($("<div></div>").text(data.title));
    for (var i=0; i<data.atom_spans.length; i++){
        span_info=data.atom_spans[i];
        
        span_text=data.contract_text.slice(span_info.start, span_info.end);
        span=$("<span></span>").text(process_text(span_text))
            .attr('data-clauses', span_info.types)
            .addClass('plain-text');
        
        two_types  = parse_clauses(span_info.types);
        // console.log(rtn)
        gold_types = two_types[0];
        pred_types = two_types[1];
        
        // Highlight gold clauses
        n_tp = gold_types.length;
        if (gold_types.length + pred_types.length == 0){
            span.addClass("span-hl-0");
        }
        else {
            span.hover(
                function() {create_info_window($(this))},
                function () {
                    if (! (pop_span_info === null)){
                        pop_span_info.addClass('disabled')
                    }
                    
                }
            );
            if (n_tp==1) {span.addClass("span-hl-1");} 
            else if (n_tp > 1) {span.addClass("span-hl-2");}
        }
        
        // Highlight prediction
        if (pred_types.length > 0) {
            is_good = false;
            for (var pti =0; pti < pred_types.length; pti++){
                r = gold_types.some(function (e){return e == pred_types[pti]});
                if (r){
                    is_good = true;
                }
            }
            if (is_good) {span.addClass("pred-correct");console.log('Good pred')}
            else {span.addClass("pred-wrong")}
        }

        cont_dom.append(span)
    }
    // cont_dom.append($("<div></div>").html(convert_html_space(data.contract_text)).addClass('plain-text'));
}

// Click callbacks

function on_click_clause_button(){
    // Update active status
    if ($(this).hasClass('active')){
        $(this).removeClass('active')
    }
    else {
        $("ul.clause-list>.list-group-item").removeClass('active')
        $(this).addClass('active')
    }
    
    // Highlight span
    if ($(this).hasClass('active')){
        cla_id = $(this).attr("data-id");
        $("#contract span").each(function () {determin_highlight($(this), cla_id)});
    }
    else {
        $("#contract span").removeClass('highlight');
    }
    // Jump to span
    first_span=$('#contract span.highlight').get(0);
    if (first_span){first_span.scrollIntoView({ block: "center" });}
    
}

function on_click_prev(){
    if (contract_id > 0){
        contract_id = contract_id - 1;
        get_contract(contract_id);
    }
}

function on_click_next(){
    get_contract(contract_id + 1);
}

function on_click_jump(){
    index = parseInt($("#ipt_index").val());
    get_contract(index);
}

function on_change_font_size(){
    fontsize = $("#font-size").val();
    $("#font-size-display").text(fontsize);
    $("#contract span").css('font-size', fontsize + 'px');
}

// Utilities

function determin_highlight(span_obj, clause_id){
    // console.log(span_obj);
    two_ids = parse_clauses(span_obj.attr('data-clauses').split(','));
    clause_types = two_ids[0];
    if (clause_types.indexOf(clause_id) > -1){
        span_obj.addClass('highlight');
    }
    else {
        span_obj.removeClass('highlight');
    }
}

function process_text(s){
    //handle space and line break
    if (! $("#show_space").is(":checked")){
        s = s.replace(/\n{2,}/g, '\n\n');
        s = s.replace(/[ ]{4,}/g, '    ')
    }

    return s
}

function set_font_size(size){
    $("#contract span").css('font-size', size);
}

// function convert_html_space(s){
//     // convert space to html label
//     s = s.replace(/\n/g, '<br>');
//     if ($("#show_space").is(":checked")){
//         s = s.replace(/[\s]{2,}/g, reg_space2html);
//     }
//     return s
// }

// function reg_space2html(match){
//     return '&nbsp'.repeat(match.length);
// }

// Pop window
function create_info_window(span_obj){
    if (pop_span_info === null){
        pop_span_info = $("<div></div>").attr('id', 'span-info');
    }
    pop_span_info.empty().removeClass('disabled');
    $("body").append(pop_span_info)
    
    
    cla_ids = span_obj.attr('data-clauses').split(',');
    two_ids = parse_clauses(cla_ids);
    gold_ids = two_ids[0];
    pred_ids = two_ids[1];
    // console.log(gold_ids);
    // console.log(pred_ids);

    cla_names = $.map(gold_ids, function(v){
        r = parseInt(v) + 1;
        return $(`#clause_sel a:nth-child(${r}) span:nth-child(2)`).text()
    })
    pred_names = $.map(pred_ids, function(v){
        r = parseInt(v) + 1;
        return $(`#clause_sel a:nth-child(${r}) span:nth-child(2)`).text()
    })
    if (cla_names.length > 0) {
        pop_span_info.append(
            sub_pop_window("Gold", cla_names, "sub-win-gold")
        );
    }
    
    if (pred_names.length > 0){
        pop_span_info.append(
            sub_pop_window("Pred", pred_names, "sub-win-pred")
        );
    }

    // console.log(pop_span_info.outerWidth());

    pop_span_info.css({
        // 'position': "fixed",
        'top': span_obj.offset().top - $(window).scrollTop() - pop_span_info.outerHeight(),
        'left': span_obj.offset().left - $(window).scrollLeft() + 0.5 * (span_obj.outerWidth() - pop_span_info.outerWidth()),
    })
}

function sub_pop_window(title, array, class_n){
    sub_win = $("<div></div>").addClass(class_n);
    sub_win.append(
        $("<div></div>").append(title)
    );
    sub_win.append(
        $("<div></div>").html(array.join('<br>'))
    );
    return sub_win
}

function parse_clauses(cla_ids){
    gold_ids = [];
    pred_ids = [];
    for (var i=0; i<cla_ids.length; i++){
        s = cla_ids[i];
        
        if (s.startsWith('gold_')){
            gold_ids.push(s.slice(5, s.length))
        }
        else if (s.startsWith('pred_')){
            pred_ids.push(s.slice(5, s.length))
        }
    }
    
    return [gold_ids, pred_ids]
}