const fs = require('fs');
const { resolve } = require('path');
const exec = require('child_process').exec;

const pkgPath = resolve(__dirname, '../package.json');
const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
const originalVersion = pkg.version;

const findNthPeriodIndex = (str, nth) => {
    let index = -1;
    for (let i = 0; i < nth; i++) {
        index = str.indexOf('.');
        str = str.slice(index + 1, str.length);
    }
    return index;
};

let startPeriod;
let endPeriod;
let n = -1;

switch(process.argv[2]) {
    case '--major':
        startPeriod = findNthPeriodIndex(pkg.version, 1);
        n = parseInt(pkg.version.slice(0, startPeriod), 10);
        pkg.version = `${n+1}.0.0`;
        break;
    case '--minor':
        startPeriod = findNthPeriodIndex(pkg.version, 2);
        endPeriod = findNthPeriodIndex(pkg.version, 3);
        n = parseInt(pkg.version.slice(startPeriod + 1, endPeriod - midPeriod), 10);
        pkg.version = `${pkg.version.slice(0, midPeriod)}.${n+1}.0`;
        break;
    case '--patch':
        endPeriod = findNthPeriodIndex(pkg.version, 3);
        n = parseInt(pkg.version.slice(endPeriod + 1, pkg.version.length), 10);
        pkg.version = `${pkg.version.slice(0, endPeriod)}${n+1}`;
        break;
}

console.log(pkg.version);
const pkgStr = JSON.stringify(pkg, null, '  ');
if (pkg.version !== originalVersion) {
    console.log('writing new package.json');
    fs.writeFileSync(pkgPath, pkgStr);
}
//console.log(pkgStr);


exports.tagExists = (tag, callback) => {
    exec(`git tag`, (err, tagsStr) => {
        if (err) {
            callback(err);
            return;
        }

        const tags = tagsStr.split('\n');
        callback(null, !!tags.filter((tagLine) => tagLine === tag).length);
    });
};

exports.tag = (callback = () =>{}) => {
    exec(`git tag v${pkg.version}`, callback);
};


exports.ensureTag = (callback = () =>{}) => {
    exports.tagExists(`v${pkg.version}`, (err, exists) => {
        if (err) {
            callback(err);
            return;
        }
        if (exists) {
            callback(null, { existed: true })
        } else {
            exports.tag((err) => {
                if (err) {
                    callback(err);
                    return;
                }
                callback(null, { existed: false });
            });
        }
    });
};
