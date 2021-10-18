'use strict';

// Get all product review links
var a = document.querySelectorAll("[data-hook='see-all-reviews-link-foot']")[0]

//If we found a link to a page of reviews
if(null != a)
{
    a.target = "_self";
    a.click();
}
//Otherwise, it means that the reviews were from Amazon.com and we don't want them.
//  Go back to listing page.
else
{
    chrome.runtime.sendMessage({message: "GetTabId"}, function (response)
    {
        var regex           = /http(s?):\/\/www\.(amazon\.[a-z]{2,3})\//;
        var regex_out       = regex.exec(window.location.href);
        var listingUrlKey   = regex_out[2] + ":" + response.windowId + ":" + response.tabId + ":listingURL";
        chrome.storage.local.get([listingUrlKey], function (result)
        {
            if(result.hasOwnProperty(listingUrlKey) && null != result[listingUrlKey])
            {
                window.location = result[listingUrlKey];
            }
        });
    });
}
