'use strict';

document.body.onload = SaveReviewsAndMoveOnToNextPage;

function SaveReviewsAndMoveOnToNextPage()
{
    // Get subsidiary
    var regex           = /http(s?):\/\/www\.(amazon\.[a-z]{2,3})\//;
    var subsidiary      = regex.exec(window.location.href)[2]

    //--------------- LOOK FOR ERRORS ---------------//
    // Check if we're getting "Page Not Found" error (along with message for API...)
    //      If not amazon.cn, just look for "Amazon.*"
    if("amazon.cn" != subsidiary)
    {
        if(!document.title.toLowerCase().startsWith(subsidiary.toLowerCase()))
        {
            alert("ERROR: No reviews");
            return;
        }
    }
    //      If amazon.cn, look for the Simplified Characters representing amazon (亚马逊, [20122, 39532, 36874])
    else if(20122 != document.title.charCodeAt(0) || 39532 != document.title.charCodeAt(1) || 36874 != document.title.charCodeAt(2))
    {
        alert("ERROR: No reviews");
        return;
    }

    // Check if we're getting "There was a problem filtering reviews..." error
    var div_error = document.getElementsByClassName("cr-error a-alert-error");
    for(var div of div_error)
    {
        //If div.offsetParent != null, then an error message is being displayed
        //  and we need to stop.
        if(div.offsetParent)
        {
            alert("ERROR: No reviews");
            return;
        }
    }
    //-----------------------------------------------//
    
    // Page number
    var pageNumber  = /pageNumber=([0-9]+)/.exec(window.location.href);
    if(null == pageNumber)
        pageNumber      = 1;
    else
        pageNumber      = parseInt(pageNumber[1]);

    // Get window and tab ids for listingURL
    chrome.runtime.sendMessage({message: "GetTabId"}, function (response)
    {
        var listingUrlKey   = subsidiary + ":" + response.windowId + ":" + response.tabId + ":listingURL";

        // Get ListingURL; include it in HTML and use it if we've run out of reviews or if we have reached page 10
        chrome.storage.local.get([listingUrlKey], function (result)
        {
            //HTML to save
            //  Add listing info at the top
            var html    = "<!--\n"
                        + "listing = " + result[listingUrlKey] + "\n"
                        + "-->\n"
                        + document.documentElement.outerHTML;
            
            //Send review html to background download page
            chrome.runtime.sendMessage({message: "download", href: window.location.href, html: html}, function (response)
            {
                // Find button to next page location
                var span        = document.querySelectorAll("[data-action='reviews:page-action']")[0];
                var li          = [];
                if(null != span)
                {
                    li              = Array.from(span.getElementsByTagName("li"));
                    li              = li.filter(function (elem)
                    { 
                        return "a-last" == elem.className
                    });
                }
                //Last page of reviews or 10th page, stop and move on
                if(0 == li.length || 10 == pageNumber)
                {
                    if(result.hasOwnProperty(listingUrlKey) && null != result[listingUrlKey])
                    {
                        window.location = result[listingUrlKey];
                    }
                }
                //Otherwise, wait for the page to finish loading, download reviews and move on to next page of reviews
                else
                {
                    //Move on to next review page
                    li[0].children[0].target = "_self";
                    li[0].children[0].click();
                    //Wait for it to load and start over
                    setTimeout(SaveReviewsAndMoveOnToNextPage, 5000);
                }
            });
        });
    });
}
