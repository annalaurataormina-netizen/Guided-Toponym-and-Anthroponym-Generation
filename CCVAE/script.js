async function loadLanguages(elementId) {
    const response = await fetch("/api/languages");
    const languages = await response.json();
    const length = languages.languages.length;
    let options = "";
    for (let idx = 0; idx < length; idx++) {
        options += "<option>" + languages.languages[idx] + "</option>";
    }
    document.getElementById(elementId).innerHTML = options;
}

const limit = 100

window.onload = function () {
    loadLanguages("languages");
    loadLanguages("languagesOther");
    document.getElementById("languagesOther").style.display = "none";
    document.getElementById("weight").style.display = "none";
};

async function generate() {
    document.getElementById("numberError").innerHTML = "";
    document.getElementById("weightError").innerHTML = "";
    document.getElementById("results").innerHTML = "Generating..."
    document.getElementById("generateBtn").disabled = true;
    const checkbox = document.getElementById("blendCheckbox");
    checkbox.disabled = true;
    try {
        const number = document.getElementById("number").value;
        const weight = document.getElementById("weight").value;
        if (number !== "" && (number < 0 || number > limit)) {
            document.getElementById("numberError").innerHTML = "Enter a number between 0 and " + limit + ".";
            document.getElementById("results").innerHTML = ""
            return;
        }
        if (weight !== "" && (weight < 0 || weight > 1)) {
            document.getElementById("weightError").innerHTML = "Enter a number between 0 and 1.";
            document.getElementById("results").innerHTML = ""
            return;
        }
        const languageDropdown = document.getElementById("languages");
        const language = languageDropdown.options[languageDropdown.selectedIndex].text;
        let url = "/api/languages/" + language;
        url += "/names"
        if (checkbox.checked) {
            const languageOtherDropdown = document.getElementById("languagesOther");
            const languageOther = languageOtherDropdown.options[languageOtherDropdown.selectedIndex].text;
            url += "?blend_language=" + languageOther;
            if (weight !== "") {
                url += "&weight=" + weight;
            }
        }
        if (number) {
            url += (url.includes("?") ? "&" : "?") + "number=" + number;
        }
        const response = await fetch(url);
        const results = await response.json();
        const length = results.names.length;
        let names = ""
        for (let idx = 0; idx < length; idx++) {
            names += "<li>" + results.names[idx] + "</li>";
        }
        document.getElementById("results").innerHTML = names;
    } catch (error) {
        document.getElementById("results").innerHTML = "Something went wrong. Please try again.";
    } finally {
        document.getElementById("generateBtn").disabled = false;
        document.getElementById("blendCheckbox").disabled = false;
    }
}

function blend() {
    const checkbox = document.getElementById("blendCheckbox");
    if (checkbox.checked) {
        document.getElementById("languagesOtherLabel").innerHTML = "Choose another language:";
        document.getElementById("languagesOther").style.display = "inline-block";
        document.getElementById("weight").style.display = "inline-block";
        document.getElementById("weightLabel").innerHTML = "Weight:";
        document.getElementById("weight").value = 0.5;
        updateWeightLabel()
        const number = document.getElementById("number").value;
        if (number !== "" && (number < 0 || number > limit)) {
            document.getElementById("number").value = "";
        }
    } else {
        document.getElementById("languagesOtherLabel").innerHTML = "";
        document.getElementById("languagesOther").style.display = "none";
        document.getElementById("weight").style.display = "none";
        document.getElementById("weightLabel").innerHTML = "";
        document.getElementById("weight").value = "";
        document.getElementById("weightNum").innerHTML = "";
        const number = document.getElementById("number").value;
        if (number !== "" && (number < 0 || number > limit)) {
            document.getElementById("number").value = "";
        }
    }
    document.getElementById("numberError").innerHTML = "";
    document.getElementById("weightError").innerHTML = "";
}

function updateWeightLabel() {
    document.getElementById("weightNum").innerHTML = document.getElementById("weight").value;
}