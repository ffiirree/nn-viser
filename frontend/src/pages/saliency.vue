<template>
    <div class="page" v-loading="loading" element-loading-background="rgba(0, 0, 0, 0.45)">
        <div class="menu">
            <div class="item">
                <div class="title">model</div>
                <el-select class="value" size="small" v-model="params.model" filterable @change="update">
                    <el-option v-for="model in models" :key="model" :value='model'/>
                </el-select>
                </div>
            <div class="item">
                <div class="title">input</div>
                <el-select class="value" size="small" v-model="params.input" @change="params.target = images[params.input]">
                    <el-option v-for="image in Object.keys(images)" :key="image" :value='image'/>
                </el-select>
            </div>
            <div class="item"><div class="title">target</div><el-input class="value" type='number' size="small" v-model="params.target"  @change="update"/></div>
            <div class="item"><div class="title"></div><el-button icon='el-icon-refresh' type="primary" size="small" circle  @click="update"/></div>
        </div>
        <div class="network">
            <div class="network-inner">
                <div class="input">
                    <div class="image-wrapper">
                        <el-image class="pixelated" :src="params.input">
                            <div slot="error" class="image-slot">
                                <i class="el-icon-lollipop"></i>
                            </div>
                        </el-image>
                        <div class="caption">Input</div>
                    </div>
                </div>
                <div class="sliency">
                    <div class="image-wrapper" v-for="key in Object.keys(saliency)" :key="key">
                        <el-image class="pixelated" :src="saliency[key]">
                            <div slot="error" class="image-slot">
                                <i class="el-icon-lollipop"></i>
                            </div>
                        </el-image>
                        <div class="caption">{{key}}</div>
                    </div>
                </div>
                <div class="sliency">
                    <div class="image-wrapper" v-for="key in Object.keys(guided_saliency)" :key="key">
                        <el-image class="pixelated" :src="guided_saliency[key]">
                            <div slot="error" class="image-slot">
                                <i class="el-icon-lollipop"></i>
                            </div>
                        </el-image>
                        <div class="caption">{{key}}</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</template>

<script>
export default {
    name: 'saliency',
    data() {
        return {
            models: [],
            images: {},
            params: {
                model: 'vgnet_g_1_0mp',
                input: '',
                target: null
            },
            saliency: {},
            guided_saliency: {},
            loading: false
        };
    },
    created() {
        this.config()
    },
    sockets: {
        response_saliency(data) {
            this.saliency = data
            this.done()
        },
        response_guided_saliency(data) {
            this.guided_saliency = data
            this.done()
        },

        models(data) {
            this.models = data
        },
        images(data) {
            this.images = data

            this.params.input = Object.keys(data)[0]
            this.params.target = this.images[this.params.input]
        }
    },
    methods: {
        config() {
            this.$socket.emit('get_models')
            this.$socket.emit('get_images')
        },
        update() {
            this.loading = true
            this.saliency = {}
            this.guided_saliency = {}
            
            this.$socket.emit("saliency", this.params);
            this.$socket.emit("guided_saliency", this.params);
        },
        done() {
            if(Object.keys(this.saliency).length && Object.keys(this.guided_saliency).length)
                this.loading = false
        }
    }
};
</script>

<style rel="stylesheet/scss" lang="scss" scoped>
.page {
    .network-inner {
        display: flex;
        flex-flow: row;

        align-items: center;
        justify-items: center;

        .input {
            flex: 0 0 auto;
        }

        .sliency {
            flex: 1 1 auto;
            display: flex;
            flex-flow: column;
            align-items: center;
            justify-items: center;
        }
    }
}
</style>
