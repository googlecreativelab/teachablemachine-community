const fs = require('fs');
const { resolve } = require('path');

const startPattern = "{{[ ]{0,3}";
const endPattern = "[ ]{0,3}}}";

const tagCapturePattern = "([A-Za-z0-9]{1,})+";

exports.writeSync = (outputPath) => {
    const code = fs.readFileSync(resolve(__dirname, './snippet.tmpl'), 'utf8');

    console.log(typeof(code));
    // const re = new RegExp(startPattern + tagCapturePattern + endPattern, 'g');

    // find all of the template tags that exist in the snippet
    const tags = Array.from(code.matchAll(new RegExp(startPattern + tagCapturePattern + endPattern, 'g')))
        .reduce((mem, match) => {
            const value = match[1];
            if (mem.indexOf(value) === -1) {
                mem.push(value);
            }
            return mem;
        }, []);

    const json = {
        startPattern,
        endPattern,
        tagCapturePattern,
        tags,
        code
    };

    const jsonStr = JSON.stringify(json, null, '    ');

    fs.writeFileSync(outputPath, jsonStr);
};
