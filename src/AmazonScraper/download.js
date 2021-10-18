'use strict';

//Save review HTML
chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse)
    {
        switch(request.message.toLowerCase())
        {
        case "gettabid":
            GetTabId(request, sender, sendResponse);
            break;
        case "download":
            DownloadPage(request, sender, sendResponse);
            break;
        }
    }
);

function GetTabId(request, sender, sendResponse)
{
    sendResponse({"windowId": sender.tab.windowId, "tabId": sender.tab.id});
}
function DownloadPage(request, sender, sendResponse)
{
    //Output directory for communicating with Java-Robot program
    const output_directory = 'AmazonScraper/reviews';

    //Build filename
    var pattern     = /http(s?):\/\/www\.(amazon\.[a-z]{2,3})\/[a-zA-Z0-9\-%]*\/product-reviews\/([A-Z0-9]*)\//;
    var regex_out   = pattern.exec(request.href);
    console.log(request.href);
    console.log(regex_out);
    var subsidiary  = regex_out[2];
    var productId   = regex_out[3];
    var pageNumber  = /pageNumber=([0-9]+)/.exec(request.href);

    if(null == pageNumber)
        pageNumber = "1";
    else
        pageNumber = pageNumber[1];

    pageNumber  = pageNumber.padStart(3, "0");
    console.log(pageNumber);

    //filename
    var filename = output_directory + '/' + subsidiary + " - " + productId + " - p" + pageNumber + ".html";

    //data to save
    var blob = new Blob([request.html], {type: "text/html"});

    //download
    var blob_url = window.URL.createObjectURL(blob);
    console.log(blob_url);
    chrome.downloads.download({
        url: blob_url,
        filename: filename,
        saveAs: false,
    }, success);
}
function success()
{
    console.log('Success!');
}