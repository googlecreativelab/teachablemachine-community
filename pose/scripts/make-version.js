// Copyright 2018 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================


// Run this script from the base directory (not the script directory):
// ./scripts/make-version

const fs = require('fs');
const { join } = require('path');
const version = JSON.parse(fs.readFileSync('package.json', 'utf8')).version;

const versionCode =
`/** @license See the LICENSE file. */

// This code is auto-generated, do not modify this file!
const version = '${version}';
export { version };

`

const defaultCallback = err => {
    if (err) {
        throw new Error(`Could not save version file ${version}: ${err}`);
    }
    console.log(`Version file for version ${version} saved sucessfully.`);
};

const write = (callback = defaultCallback) =>
    fs.writeFile(join(__dirname, '../src/version.ts'), versionCode, callback);


if (process.argv[1] && process.argv[1].indexOf('make-version') > -1) {
    // it was run as a command, so self-execute
    write();
}

exports.write = write;
