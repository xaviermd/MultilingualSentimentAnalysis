'use strict';

// Get links to products with reviews
var span        = document.querySelectorAll("[data-component-type='s-search-results']")[0];
var regex       = /http(s?):\/\/www\.amazon\.[a-z]{2,3}\/([a-zA-Z0-9%\-]+\/)?dp\/([A-Z0-9]*)\//;
var links       = null;
var nextPage    = null;
//  Listing
if(null != span)
{
    links   = Array.from(span.getElementsByClassName("a-link-normal"));
    links = links.filter(function (link)
    { 
        return 1 == link.classList.length
            && 1 == link.childElementCount
            && "span" == link.children[0].tagName.toLowerCase()
            && 1 == link.children[0].classList.length
            && "a-size-base" == link.children[0].classList[0]
            && regex.test(link.href);
    })

    // Find button to next page
    nextPage    = document.getElementsByClassName('a-normal');
    if(!/&page=([0-9]*)/.test(document.location.href) || 1 == /&page=([0-9]*)/.exec(document.location.href)[1])
    {
        //If page 1, get first button
        nextPage        = nextPage[0];
    }
    else
    {
        //Otherwise, get last button
        nextPage        = nextPage[nextPage.length-1];
    }
    nextPage    = nextPage.children[0]
}
else
//  Featured
{
    span = document.getElementById("mainResults");
    links = Array.from(span.getElementsByClassName("a-size-small a-link-normal a-text-normal"));
    regex   = /http(s?):\/\/www\.amazon\.[a-z]{2,3}\/([a-zA-Z0-9%\-]+\/)?dp\/([A-Z0-9]*)\//;
    links = links.filter(function (link)
    { 
        return regex.test(link.href);
            //&& 0 == link.childElementCount;
    })

    // Find button to next page
    nextPage = document.getElementsByClassName('pagnLink');
    if(!/&page=([0-9]*)/.test(document.location.href) || 1 == /&page=([0-9]*)/.exec(document.location.href)[1])
        //If page 1, get first button
        nextPage = nextPage[0];
    else
        //Otherwise, get last button
        nextPage = nextPage[nextPage.length-1];
    nextPage = nextPage.children[0];
}

// Prepare productId/HTML link element dictionary
var dict    = {}
var regex_out = null;
for(var link of links)
{
    regex_out     = regex.exec(link.href);
    dict[regex_out[regex_out.length-1]]
                = link;
}

chrome.runtime.sendMessage({message: "GetTabId"}, function (response)
{
    var regex           = /http(s?):\/\/www\.(amazon\.[a-z]{2,3})\//;
    var regex_out       = regex.exec(window.location.href);
    var productIdKey    = regex_out[2] + ":productIds";
    var listingUrlKey   = regex_out[2] + ":" + response.windowId + ":" + response.tabId + ":listingURL";
    var to_save         = {};

    // Get productIds which have already been scraped
    chrome.storage.local.get([productIdKey], function (result)
    {
        var already = result[productIdKey];
        if(!already)
            already = [];

        // Remove productIds (and corresponding links) that we have already scraped
        for(var i = 0; i < already.length; i++)
        {
            delete dict[already[i]];
        }

        //Get remaining product links
        var id  = Object.keys(dict);
        
        //  If there are still products remaining to be scraped
        console.log(id.length);
        if(0 < id.length)
        {
            //Pick link at random
            id      = id[Math.floor(Math.random() * id.length)];

            //Add link to list of scraped productIds and save the new array
            already.push(id);
            to_save[productIdKey]   = already;
            to_save[listingUrlKey]  = window.location.href;
            console.log(listingUrlKey);
            console.log(to_save);
            chrome.storage.local.set(to_save);
            
            //Click link
            dict[id].target = "_self";
            //dict[id].click();
            setTimeout(function(){ dict[id].click(); }, 5000);
        }
        else
        {
            //Get the page number
            var pageNumber = /page=([0-9]+)/.exec(window.location.href);
            //Stop at page 10.
            if(null == pageNumber || 10 > parseInt(pageNumber[1]))
            {
                //Next page
                nextPage.target = "_self";
                nextPage.click();
            }
            else
            {
                alert('Done')
            }
        }
    });
});
